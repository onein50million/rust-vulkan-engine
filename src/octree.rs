pub(crate) mod node_types {
    pub(crate) const EMPTY: u32 = 0;
    pub(crate) const CHILDREN: u32 = 1;
    pub(crate) const REFLECTIVE: u32 = 2;
}

const TREE_DEPTH: usize = 10;

#[derive(Copy, Clone)]
#[repr(C)]
pub(crate) struct Node {
    pub(crate) node_type: u32,
    pub(crate) leaf_indices: [u32; 8],
    pub(crate) next_deeper_node: u32,
    pub(crate) next_same_node: u32,
    pub(crate) parent_node: u32,
    pub(crate) current_depth: u32,
    pub(crate) path: [u32; TREE_DEPTH],
}
impl Node {
    pub(crate) fn new(node_type: u32, node_indices: [u32; 8]) -> Self {
        return Node {
            node_type,
            leaf_indices: node_indices,
            next_deeper_node: 0,
            next_same_node: 0,
            parent_node: 0,
            current_depth: 0,
            path: [0; TREE_DEPTH],
        };
    }
}

pub(crate) struct Octree {
    pub(crate) nodes: Vec<Node>,
}
impl Octree {
    pub(crate) fn new() -> Self {
        let mut nodes: Vec<Node> = vec![];
        nodes.push(Node {
            node_type: node_types::EMPTY,
            leaf_indices: [0; 8],
            next_deeper_node: 0,
            next_same_node: 0,
            parent_node: 0,
            current_depth: 0,
            path: [0; TREE_DEPTH],
        });
        let octree = Octree { nodes };
        return octree;
    }
    fn add_node(&mut self, parent_index: u32, branch_index: u32, node_type: u32) -> u32 {
        let mut new_node = Node::new(node_type, [0; 8]);
        new_node.parent_node = parent_index;
        new_node.current_depth = self.nodes[parent_index as usize].current_depth + 1;
        new_node.path = self.nodes[parent_index as usize].path;
        new_node.path[(new_node.current_depth - 1) as usize] = branch_index;
        assert_ne!(
            self.nodes[parent_index as usize].node_type,
            node_types::REFLECTIVE
        );
        self.nodes[parent_index as usize].node_type = node_types::CHILDREN;
        assert_eq!(
            self.nodes[parent_index as usize].leaf_indices[branch_index as usize],
            0
        ); //make sure branch isn't already assigned
        self.nodes.push(new_node);
        let new_index = (self.nodes.len() - 1) as u32;
        self.nodes[parent_index as usize].leaf_indices[branch_index as usize] = new_index;
        return new_index;
    }
    pub(crate) fn fill_random(&mut self, depth: u32) {
        let rng = fastrand::Rng::new();
        let mut nodes_with_children_indices = vec![0];
        let mut current_depth = 0;
        while current_depth <= depth {
            let mut new_nodes_with_children_indices = vec![];
            for node_with_children_index in nodes_with_children_indices {
                for branch in 0..8 {
                    if rng.bool() {
                        new_nodes_with_children_indices.push(self.add_node(
                            node_with_children_index,
                            branch,
                            node_types::EMPTY,
                        ));
                    } else if rng.bool() {
                        self.add_node(node_with_children_index, branch, node_types::REFLECTIVE);
                    }
                }
            }
            nodes_with_children_indices = new_nodes_with_children_indices;
            current_depth += 1;
        }

        // let mut nodes_with_children_indices = vec![0];
        // let mut current_depth = 0;
        // while current_depth < depth{
        //     let mut new_nodes_with_children_indices = vec![];
        //     for node_with_children_index in nodes_with_children_indices{
        //         for branch in 0..8{
        //             if current_depth == 0{
        //                 if branch % 2 == 1 || ((branch/2) % 2) == 0 {
        //                     new_nodes_with_children_indices.push(self.add_node(node_with_children_index, branch, node_types::EMPTY));
        //                 }
        //             }
        //             else if current_depth == depth - 1  {
        //                 new_nodes_with_children_indices.push(self.add_node(node_with_children_index, branch, node_types::EMPTY));
        //             }
        //
        //
        //         }
        //     }
        //     nodes_with_children_indices = new_nodes_with_children_indices;
        //     current_depth += 1;
        // }
    }

    fn get_next_deeper_node(&self, node_index: u32, mut visited: Vec<u32>) -> (u32, Vec<u32>) {
        if !visited.contains(&node_index) {
            visited.push(node_index);
        }
        let mut current_index = node_index;

        loop {
            for current_branch in 0..8 {
                let next_node_index =
                    self.nodes[current_index as usize].leaf_indices[current_branch as usize];
                if next_node_index != 0
                    && !visited.contains(&next_node_index)
                    && self.nodes[next_node_index as usize].node_type == node_types::CHILDREN
                {
                    visited.push(next_node_index);
                    return (next_node_index, visited);
                }
            }
            if current_index == 0 {
                return (0, visited);
            }
            current_index = self.nodes[current_index as usize].parent_node;
        }
    }
    fn get_next_same_node(&self, node_index: u32, mut visited: Vec<u32>) -> (u32, Vec<u32>) {
        if !visited.contains(&node_index) {
            visited.push(node_index);
        }
        if node_index == 0 {
            return (0, visited);
        }
        let mut current_index = self.nodes[node_index as usize].parent_node;

        loop {
            for current_branch in 0..8 {
                let next_node_index =
                    self.nodes[current_index as usize].leaf_indices[current_branch as usize];
                if next_node_index != 0
                    && !visited.contains(&next_node_index)
                    && self.nodes[next_node_index as usize].node_type == node_types::CHILDREN
                {
                    visited.push(next_node_index);
                    return (next_node_index, visited);
                }
            }
            if current_index == 0 {
                return (0, visited);
            }
            current_index = self.nodes[current_index as usize].parent_node;
        }
    }

    pub(crate) fn calculate_nexts(&mut self) {
        let mut current_node_index = 0;
        let mut next_visited = vec![];
        let mut same_visited = vec![];

        loop {
            let (next_node, updated_next_visited) =
                self.get_next_deeper_node(current_node_index, next_visited);
            next_visited = updated_next_visited;
            assert_eq!(
                self.nodes[next_node as usize].node_type,
                node_types::CHILDREN
            );
            let (next_same_level_node, updated_same_visited) =
                self.get_next_same_node(current_node_index, same_visited);
            assert_eq!(
                self.nodes[next_same_level_node as usize].node_type,
                node_types::CHILDREN
            );
            same_visited = updated_same_visited;

            if next_same_level_node != 0 {
                self.nodes[current_node_index as usize].next_same_node = next_same_level_node;
            }
            if next_node == 0 {
                break;
            }
            self.nodes[current_node_index as usize].next_deeper_node = next_node;
            current_node_index = next_node;
        }
    }
}
