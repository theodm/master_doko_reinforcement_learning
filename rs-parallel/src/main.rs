use drop_arena::{DropArena, DropBox};

mod Buffer2;

struct UnsafeArena<T> {
    buffer: Vec<T>,

    free_indices_list: Vec<usize>
}


impl <T> UnsafeArena<T> {
    fn new(capacity: usize) -> Self {
        let freelist = (0..capacity).collect();

        let mut vec = Vec::with_capacity(capacity);

        unsafe {
            vec.set_len(capacity);
        }

        Self {
            buffer: vec,
            free_indices_list: freelist
        }
    }

    fn alloc(&mut self, value: T) -> *mut T {
        match self.free_indices_list.pop() {
            Some(index) => {
                self.buffer[index] = value;

                &mut self
                    .buffer[index] as *mut T
            },
            None => {
                self.buffer.push(value);

                let new_len = self.buffer.len();

                &mut self
                    .buffer[new_len - 1] as *mut T
            }
        }
    }

    fn free_all(&mut self) {
        self.free_indices_list.clear();

        for i in 0..self.buffer.len() {
            self.free_indices_list.push(i);
        }
    }

    fn free_single(&mut self, index: usize) {
        self.free_indices_list.push(index);
    }
}

#[derive(Debug, Clone)]
struct Node {
    children: heapless::Vec<*mut Node, 3>,
    parent: *mut Node,

    value: usize
}

fn main() {
    let mut arena = UnsafeArena::<Node>::new(10);

    let mut root = Node {
        children: heapless::Vec::new(),
        parent: std::ptr::null_mut(),
        value: 0
    };

    let root_ptr = arena.alloc(root);

    let mut child = Node {
        children: heapless::Vec::new(),
        parent: root_ptr,
        value: 1
    };

    let child_ptr = arena.alloc(child);

    // Set child as child of root
    unsafe {
        let root = &mut *root_ptr;

        root.children.push(child_ptr).unwrap();
    }

    println!("Root: {:?}", unsafe { &*root_ptr });
    // println!("Child: {:?}", unsafe { (*(*root_ptr).children[0]) });


}
