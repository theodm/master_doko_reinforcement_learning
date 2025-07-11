
pub struct UnsafeArena<T> {
    buffer: Vec<T>,

    free_indices_list: Vec<usize>
}

impl <T> UnsafeArena<T> {
    pub fn new(capacity: usize) -> Self {
        let freelist = (0..capacity).collect();

        let mut vec: Vec<T> = Vec::with_capacity(capacity);

        unsafe {
            vec.set_len(capacity);

            // Allocate memory for the buffer
            for i in 0..capacity {
                std::ptr::write(vec.as_mut_ptr().add(i), std::mem::zeroed());
            }
        }

        Self {
            buffer: vec,
            free_indices_list: freelist
        }
    }

    pub fn alloc_with_index(&mut self, value: T) -> (usize, *mut T) {
        match self.free_indices_list.pop() {
            Some(index) => {
                self.buffer[index] = value;

                (index, &mut self
                    .buffer[index] as *mut T)
            },
            None => {
                panic!("No more space in the arena: {}", self.buffer.len());
            }
        }
    }

    pub fn alloc(&mut self, value: T) -> *mut T {
        return self.alloc_with_index(value).1;
    }

    pub fn free_all(&mut self) {
        // let not_free_indices_len = self.buffer.len() - self.free_indices_list.len();

        // println!("Freeing {} indices", not_free_indices_len);

        self.free_indices_list.clear();

        for i in 0..self.buffer.len() {
            self.free_indices_list.push(i);
        }
    }

    pub fn free_single(&mut self, index: usize) {
        self.free_indices_list.push(index);
    }
}