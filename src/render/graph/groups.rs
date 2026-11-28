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

use std::cell::UnsafeCell;
use std::sync::atomic::AtomicUsize;

use super::Graph;
use crate::context::AudioNodeId;
use crate::render::{AudioRenderQuantum, NodeCollection, QuantumCopy};

/// A value that one thread at a time works on, in the order the render schedule gives
///
/// Mailboxes and drop lists sit in these, so render threads can share the plan while each works
/// on its own group.
#[derive(Debug, Default)]
pub(in crate::render) struct SyncCell<T>(UnsafeCell<T>);

// SAFETY: every caller of `get` upholds that one thread at a time works on the value, ordered by
// the schedule's atomics and work queue (see `Workers`)
unsafe impl<T: Send> Sync for SyncCell<T> {}

impl<T> SyncCell<T> {
    pub fn new(value: T) -> Self {
        Self(UnsafeCell::new(value))
    }

    pub fn get_mut(&mut self) -> &mut T {
        self.0.get_mut()
    }

    /// A shared look, for tests that hold the graph whole
    #[cfg(test)]
    pub fn get_ref(&self) -> &T {
        // SAFETY: a test reads the plan only while nothing renders
        unsafe { &*self.0.get() }
    }

    /// # Safety
    ///
    /// No other reference to the value may be alive. During a quantum, only the thread that
    /// renders the group the value belongs to may call this.
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn get(&self) -> &mut T {
        // SAFETY: guaranteed by the caller
        unsafe { &mut *self.0.get() }
    }
}

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
        nodes.wake(self.to);
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
    /// Nodes that ended this quantum, removed once every group rendered
    pub dropped: SyncCell<Vec<AudioNodeId>>,
}

/// How the graph splits into render groups, rebuilt with the node order
#[derive(Debug, Default)]
pub(super) struct GroupPlan {
    /// Empty when every node is in render group 0
    pub groups: Vec<Group>,
    /// Group indices in dependency order
    pub order: Vec<usize>,
    /// One per edge between groups
    pub mailboxes: Vec<SyncCell<Mailbox>>,
    /// Per group, the groups it still waits for in this quantum
    pub pending: Vec<AtomicUsize>,
    /// The group of each node, by node id, `usize::MAX` for none
    node_group: Vec<usize>,
    /// The group of each render group id, sorted by id
    render_groups: Vec<(u32, usize)>,
    /// Mailboxes when the plan was last built, so patched ones cannot pile up
    built_mailboxes: usize,
    /// Groups whose list a patch reorders, and the run it reorders
    stale: Vec<bool>,
    run: Vec<AudioNodeId>,
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
            plan.render_groups.clear();
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
        plan.pending.resize_with(count, AtomicUsize::default);
        for group in &mut plan.groups {
            group.ordered.clear();
            group.incoming.clear();
            group.dependents.clear();
            group.dependencies = 0;
            group.dropped.get_mut().clear();
        }
        for (v, &id) in ids.iter().enumerate() {
            if find(parent, v) == v {
                plan.groups[plan.slot[v]].id = id;
            }
        }

        plan.render_groups.clear();
        plan.render_groups
            .extend(ids.iter().enumerate().map(|(v, &id)| (id, plan.slot[v])));

        // A filtered topological order is still topological within the group
        plan.node_group.clear();
        plan.node_group.resize(self.nodes.id_count(), usize::MAX);
        for &id in &self.ordered {
            let group = plan.slot[index(self.nodes.get_unchecked(id).borrow().render_group)];
            plan.groups[group].ordered.push(id);
            plan.node_group[id.0 as usize] = group;
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
                        let old = old.get_mut();
                        let quantum = std::mem::take(&mut old.quantum);
                        *old = Mailbox { quantum, ..mailbox };
                    }
                    None => plan.mailboxes.push(SyncCell::new(mailbox)),
                }
                plan.groups[to].incoming.push(mailboxes);
                edge.mailbox = mailboxes;
                mailboxes += 1;
            }
        }
        plan.mailboxes.truncate(mailboxes);
        plan.built_mailboxes = mailboxes;
        let mailboxes = &mut plan.mailboxes;
        for group in &mut plan.groups {
            group.incoming.sort_unstable_by_key(|&m| {
                let m = mailboxes[m].get_mut();
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

    /// Gives a new node with no edges the end of its group's list. False when its render
    /// group has no group in the plan, which a rebuild makes.
    pub(super) fn place_in_plan(&mut self, id: AudioNodeId, render_group: u32) -> bool {
        let plan = &mut *self.plan;
        let Ok(found) = plan
            .render_groups
            .binary_search_by_key(&render_group, |&(group, _)| group)
        else {
            return false;
        };
        let group = plan.render_groups[found].1;
        let index = id.0 as usize;
        if plan.node_group.len() <= index {
            plan.node_group.resize(index + 1, usize::MAX);
        }
        plan.node_group[index] = group;
        plan.groups[group].ordered.push(id);
        true
    }

    /// Patches the plan for an edge the order took without a re-sort (`repair_order`)
    ///
    /// The lists of the groups whose nodes moved are filtered from the order again. An edge
    /// between groups already linked gets a mailbox, as a rebuild would make it. False when
    /// only a rebuild will do: the edge is a param's into another group, which joins the
    /// two, or it links two groups that were not, which may join them in a cycle.
    pub(super) fn plan_edge(
        &mut self,
        source: (AudioNodeId, usize),
        dest: (AudioNodeId, usize),
    ) -> bool {
        let plan = &mut *self.plan;
        if plan.groups.is_empty() {
            return true;
        }
        let group_of = |plan: &GroupPlan, id: AudioNodeId| {
            plan.node_group
                .get(id.0 as usize)
                .copied()
                .filter(|&group| group != usize::MAX)
        };

        // A group's list is the order filtered to it, so the span the order reordered is a
        // run of it too, found by position; within the run its moved nodes go last, in order.
        if !self.repair.moved.is_empty() {
            plan.stale.clear();
            plan.stale.resize(plan.groups.len(), false);
            for &id in &self.repair.moved {
                let Some(group) = group_of(plan, id) else {
                    return false;
                };
                plan.stale[group] = true;
            }
            let (first, last) = self.repair.span;
            let repair = &*self.repair;
            let position = |id: AudioNodeId| repair.position(id).unwrap_or(usize::MAX);
            for (group, &stale) in plan.groups.iter_mut().zip(&plan.stale) {
                if !stale {
                    continue;
                }
                let list = &mut group.ordered;
                let start = list.partition_point(|&id| position(id) < first);
                let end = list.partition_point(|&id| position(id) <= last);
                plan.run.clear();
                plan.run.extend(
                    list[start..end]
                        .iter()
                        .copied()
                        .filter(|&id| !repair.moved_last(id)),
                );
                plan.run.extend(
                    list[start..end]
                        .iter()
                        .copied()
                        .filter(|&id| repair.moved_last(id)),
                );
                list[start..end].copy_from_slice(&plan.run);
            }
        }

        let (Some(from), Some(to)) = (group_of(plan, source.0), group_of(plan, dest.0)) else {
            return false;
        };
        if from == to {
            return true;
        }
        // ponytail: a patch never drops a mailbox; past twice the built ones, rebuild
        if dest.1 == usize::MAX
            || !plan.groups[from].dependents.contains(&to)
            || plan.mailboxes.len() >= 2 * plan.built_mailboxes + 64
        {
            return false;
        }

        let mailbox = plan.mailboxes.len();
        plan.mailboxes.push(SyncCell::new(Mailbox {
            from: source.0,
            output: source.1,
            to: dest.0,
            input: dest.1,
            written: false,
            quantum: QuantumCopy::default(),
        }));
        let mailboxes = &mut plan.mailboxes;
        let incoming = &mut plan.groups[to].incoming;
        incoming.push(mailbox);
        incoming.sort_unstable_by_key(|&m| {
            let m = mailboxes[m].get_mut();
            (m.from.0, m.output, m.to.0, m.input)
        });
        let mut node = self.nodes.get_unchecked(source.0).borrow_mut();
        let edge = node
            .outgoing_edges
            .iter_mut()
            .rev()
            .find(|edge| {
                edge.other_id == dest.0 && edge.self_index == source.1 && edge.other_index == dest.1
            })
            .expect("the edge was just added");
        edge.mailbox = mailbox;
        true
    }

    /// Panics unless the plan is what a rebuild from the current order would make: each
    /// group's list the order filtered to it, and a mailbox for each edge between groups,
    /// delivered by its consumer in producer order
    #[cfg(test)]
    pub(super) fn assert_plan_matches_order(&self) {
        let plan = &*self.plan;
        if plan.groups.is_empty() {
            return;
        }
        for (index, group) in plan.groups.iter().enumerate() {
            let expected: Vec<_> = self
                .ordered
                .iter()
                .copied()
                .filter(|id| plan.node_group[id.0 as usize] == index)
                .collect();
            assert_eq!(group.ordered, expected, "group {index}'s list");
        }
        // a node muted in a cycle has no place in the order, so its group is its render group's
        let group_of = |id: AudioNodeId| {
            let render_group = self.nodes.get_unchecked(id).borrow().render_group;
            let found = plan
                .render_groups
                .binary_search_by_key(&render_group, |&(group, _)| group)
                .expect("a render group in the plan");
            plan.render_groups[found].1
        };
        for &id in &self.ordered {
            assert_eq!(
                plan.node_group[id.0 as usize],
                group_of(id),
                "group of {id:?}"
            );
        }
        for id in self.nodes.keys() {
            let node = self.nodes.get_unchecked(id).borrow();
            let from = group_of(id);
            for edge in &node.outgoing_edges {
                let to = if edge.other_id == id {
                    from
                } else {
                    group_of(edge.other_id)
                };
                if from == to || edge.other_id == id {
                    assert_eq!(edge.mailbox, usize::MAX, "edge in one group");
                    continue;
                }
                let mailbox = plan.mailboxes[edge.mailbox].get_ref();
                assert_eq!(
                    (mailbox.from, mailbox.output, mailbox.to, mailbox.input),
                    (id, edge.self_index, edge.other_id, edge.other_index)
                );
                assert!(plan.groups[to].incoming.contains(&edge.mailbox));
                assert!(plan.groups[from].dependents.contains(&to));
            }
        }
        for group in &plan.groups {
            let keys: Vec<_> = group
                .incoming
                .iter()
                .map(|&m| {
                    let m = plan.mailboxes[m].get_ref();
                    (m.from.0, m.output, m.to.0, m.input)
                })
                .collect();
            assert!(
                keys.windows(2).all(|pair| pair[0] <= pair[1]),
                "incoming order"
            );
        }
    }
}
