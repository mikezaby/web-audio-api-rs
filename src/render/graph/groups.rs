//! Render groups: parts of the graph that render as one unit, the unit of parallel rendering
//!
//! Nodes are tagged with a render group at registration (`BaseAudioContext::set_render_group`).
//! When the node order is rebuilt, the groups are merged where they cannot render apart:
//!
//! - groups joined by a hidden edge (`other_index == usize::MAX`): the receiver reads the
//!   sender's buffer straight from the node collection. This keeps `AudioParam`s with their
//!   owner, and puts every `PannerNode` in the group of the `AudioListener`, which is group 0.
//! - groups that depend on each other in both directions, directly or through other groups (the
//!   strongly connected components of the group graph). Edges cleared by a cycle breaker are gone
//!   by then, so they do not count.
//!
//! Each render group has its own buffer pool, and every edge between two render groups copies:
//! through a mailbox between merged groups, straight into the other pool within one. So a buffer
//! never leaves its render group, however the groups merge now or split later.

use super::Graph;
use crate::context::AudioNodeId;
use crate::render::{AudioRenderQuantum, NodeCollection, QuantumCopy};

/// Carries one edge between render groups
///
/// The producer posts its output after it renders. The consumer adds it to its input before its
/// own group renders.
#[derive(Debug)]
pub(in crate::render) struct Mailbox {
    from: AudioNodeId,
    output: usize,
    to: AudioNodeId,
    input: usize,
    /// Set when the producer rendered this quantum
    written: bool,
    quantum: QuantumCopy,
}

impl Mailbox {
    pub fn post(&mut self, signal: &AudioRenderQuantum) {
        self.quantum.copy_from(signal);
        self.written = true;
    }

    pub fn deliver(&mut self, nodes: &NodeCollection) {
        if !std::mem::take(&mut self.written) {
            return;
        }

        let mut node = nodes.get_unchecked(self.to).borrow_mut();
        node.has_inputs_connected = true;
        let signal = self.quantum.to_quantum(&node.alloc);
        let channel_config = node.channel_config.clone();
        node.inputs[self.input].add(&signal, &channel_config);
    }
}

/// A merged render group
#[derive(Debug, Default)]
pub(super) struct Group {
    /// Smallest render group id among the groups merged into this one
    pub id: u32,
    /// Nodes of this group, in render order
    pub ordered: Vec<AudioNodeId>,
    /// Mailboxes to deliver before rendering, ordered by producer node id, then output index
    pub incoming: Vec<usize>,
    /// Groups that consume output of this group
    pub dependents: Vec<usize>,
    /// Number of groups this group consumes output from
    pub dependencies: usize,
}

/// How the graph splits into render groups, rebuilt with the node order
#[derive(Debug, Default)]
pub(super) struct GroupPlan {
    /// Empty when every node is in render group 0
    pub groups: Vec<Group>,
    /// Group indices in dependency order
    pub order: Vec<usize>,
    /// One per edge between groups
    pub mailboxes: Vec<Mailbox>,
    // Scratch space, kept to avoid allocating on every rebuild
    ids: Vec<u32>,
    parent: Vec<usize>,
    links: Vec<(usize, usize)>,
    slot: Vec<usize>,
    scc: Scc,
}

/// Tarjan's strongly connected components, state
#[derive(Debug, Default)]
struct Scc {
    index: Vec<usize>,
    low: Vec<usize>,
    on_stack: Vec<bool>,
    stack: Vec<usize>,
    next: usize,
}

const UNVISITED: usize = usize::MAX;

fn find(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    x
}

/// Merge two sets, the smallest index stays root so group 0 keeps id 0
fn union(parent: &mut [usize], a: usize, b: usize) {
    let (a, b) = (find(parent, a), find(parent, b));
    parent[a.max(b)] = a.min(b);
}

/// Edges leaving `v`, from `links` sorted by source
fn links_from(links: &[(usize, usize)], v: usize) -> impl Iterator<Item = usize> + '_ {
    let start = links.partition_point(|&(from, _)| from < v);
    links[start..]
        .iter()
        .take_while(move |&&(from, _)| from == v)
        .map(|&(_, to)| to)
}

impl Scc {
    fn reset(&mut self, len: usize) {
        self.index.clear();
        self.index.resize(len, UNVISITED);
        self.low.clear();
        self.low.resize(len, 0);
        self.on_stack.clear();
        self.on_stack.resize(len, false);
        self.stack.clear();
        self.next = 0;
    }

    /// Visit `v` and merge every strongly connected component found into one set
    fn connect(&mut self, v: usize, links: &[(usize, usize)], parent: &mut [usize]) {
        self.index[v] = self.next;
        self.low[v] = self.next;
        self.next += 1;
        self.stack.push(v);
        self.on_stack[v] = true;

        for w in links_from(links, v) {
            if self.index[w] == UNVISITED {
                self.connect(w, links, parent);
                self.low[v] = self.low[v].min(self.low[w]);
            } else if self.on_stack[w] {
                self.low[v] = self.low[v].min(self.index[w]);
            }
        }

        if self.low[v] == self.index[v] {
            loop {
                let w = self.stack.pop().unwrap();
                self.on_stack[w] = false;
                union(parent, v, w);
                if w == v {
                    break;
                }
            }
        }
    }
}

impl Graph {
    /// Build the group plan from the current node order and edges
    pub(super) fn plan_groups(&mut self) {
        // Keep the path without groups as cheap as it was: with no groups now and no plan before,
        // no edge carries a mailbox index
        let grouped = self
            .nodes
            .keys()
            .any(|id| self.nodes.get_unchecked(id).borrow().render_group != 0);
        if !grouped && self.plan.groups.is_empty() {
            return;
        }

        for node in self.nodes.values_mut() {
            for edge in &mut node.get_mut().outgoing_edges {
                edge.mailbox = usize::MAX;
            }
        }

        let plan = &mut *self.plan;
        plan.order.clear();

        plan.ids.clear();
        plan.ids.extend(
            self.nodes
                .keys()
                .map(|id| self.nodes.get_unchecked(id).borrow().render_group),
        );
        plan.ids.sort_unstable();
        plan.ids.dedup();
        if plan.ids.len() <= 1 {
            plan.groups.clear();
            plan.mailboxes.clear();
            return;
        }

        let ids = &plan.ids;
        let index = |group: u32| ids.binary_search(&group).unwrap();
        let parent = &mut plan.parent;
        parent.clear();
        parent.extend(0..ids.len());

        // Merge groups joined by hidden edges, and collect the other edges between groups
        plan.links.clear();
        for id in self.nodes.keys() {
            let node = self.nodes.get_unchecked(id).borrow();
            let from = index(node.render_group);
            for edge in &node.outgoing_edges {
                let to = index(
                    self.nodes
                        .get_unchecked(edge.other_id)
                        .borrow()
                        .render_group,
                );
                if edge.other_index == usize::MAX {
                    union(parent, from, to);
                } else if from != to {
                    plan.links.push((from, to));
                }
            }
        }

        // Merge groups that depend on each other in both directions
        for link in plan.links.iter_mut() {
            *link = (find(parent, link.0), find(parent, link.1));
        }
        plan.links.retain(|&(from, to)| from != to);
        plan.links.sort_unstable();
        plan.links.dedup();
        plan.scc.reset(ids.len());
        for v in 0..ids.len() {
            if plan.scc.index[v] == UNVISITED {
                plan.scc.connect(v, &plan.links, parent);
            }
        }

        // Number the merged groups by their smallest id, so the destination's group is 0
        plan.slot.clear();
        plan.slot.resize(ids.len(), UNVISITED);
        let mut count = 0;
        for v in 0..ids.len() {
            let root = find(parent, v);
            if plan.slot[root] == UNVISITED {
                plan.slot[root] = count;
                count += 1;
            }
            plan.slot[v] = plan.slot[root];
        }
        plan.groups.resize_with(count, Group::default);
        for group in &mut plan.groups {
            group.ordered.clear();
            group.incoming.clear();
            group.dependents.clear();
            group.dependencies = 0;
        }
        for (v, &id) in ids.iter().enumerate() {
            if find(parent, v) == v {
                plan.groups[plan.slot[v]].id = id;
            }
        }

        // A filtered topological order is still topological within the group
        for &id in &self.ordered {
            let group = index(self.nodes.get_unchecked(id).borrow().render_group);
            plan.groups[plan.slot[group]].ordered.push(id);
        }

        // Edges between groups go through mailboxes, reusing the previous ones
        let mut mailboxes = 0;
        for id in self.nodes.keys() {
            let mut node = self.nodes.get_unchecked(id).borrow_mut();
            let from = plan.slot[index(node.render_group)];
            for edge in node.outgoing_edges.iter_mut() {
                let to = if edge.other_id == id {
                    from
                } else {
                    let other = self.nodes.get_unchecked(edge.other_id).borrow();
                    plan.slot[index(other.render_group)]
                };
                if from == to {
                    continue;
                }

                let mailbox = Mailbox {
                    from: id,
                    output: edge.self_index,
                    to: edge.other_id,
                    input: edge.other_index,
                    written: false,
                    quantum: QuantumCopy::default(),
                };
                match plan.mailboxes.get_mut(mailboxes) {
                    Some(old) => {
                        let quantum = std::mem::take(&mut old.quantum);
                        *old = Mailbox { quantum, ..mailbox };
                    }
                    None => plan.mailboxes.push(mailbox),
                }
                plan.groups[to].incoming.push(mailboxes);
                edge.mailbox = mailboxes;
                mailboxes += 1;
            }
        }
        plan.mailboxes.truncate(mailboxes);
        for group in &mut plan.groups {
            group.incoming.sort_unstable_by_key(|&m| {
                let m = &plan.mailboxes[m];
                (m.from.0, m.output, m.to.0, m.input)
            });
        }

        for link in plan.links.iter_mut() {
            *link = (plan.slot[link.0], plan.slot[link.1]);
        }
        plan.links.retain(|&(from, to)| from != to);
        plan.links.sort_unstable();
        plan.links.dedup();
        for &(from, to) in &plan.links {
            plan.groups[from].dependents.push(to);
            plan.groups[to].dependencies += 1;
        }

        // Kahn's algorithm, `order` doubles as the queue
        plan.slot.clear();
        plan.slot
            .extend(plan.groups.iter().map(|group| group.dependencies));
        plan.order
            .extend((0..count).filter(|&group| plan.slot[group] == 0));
        let mut next = 0;
        while next < plan.order.len() {
            let group = plan.order[next];
            next += 1;
            for &dependent in &plan.groups[group].dependents {
                plan.slot[dependent] -= 1;
                if plan.slot[dependent] == 0 {
                    plan.order.push(dependent);
                }
            }
        }
        debug_assert_eq!(plan.order.len(), count, "render groups left in a cycle");
    }
}
