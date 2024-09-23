// Required traits and types for MnistImage and MnistDigit
#[derive(Debug, Clone, Copy, PartialEq)] // Derive Clone for MnistDigit
pub enum MnistDigit {
    Zero,
    One,
    Two,
    Three,
    Four,
    Five,
    Six,
    Seven,
    Eight,
    Nine,
}

impl MnistDigit {
    pub fn as_usize(&self) -> usize {
        match self {
            MnistDigit::Zero => 0,
            MnistDigit::One => 1,
            MnistDigit::Two => 2,
            MnistDigit::Three => 3,
            MnistDigit::Four => 4,
            MnistDigit::Five => 5,
            MnistDigit::Six => 6,
            MnistDigit::Seven => 7,
            MnistDigit::Eight => 8,
            MnistDigit::Nine => 9,
        }
    }

    pub fn from_usize(digit: usize) -> MnistDigit {
        match digit {
            0 => MnistDigit::Zero,
            1 => MnistDigit::One,
            2 => MnistDigit::Two,
            3 => MnistDigit::Three,
            4 => MnistDigit::Four,
            5 => MnistDigit::Five,
            6 => MnistDigit::Six,
            7 => MnistDigit::Seven,
            8 => MnistDigit::Eight,
            9 => MnistDigit::Nine,
            _ => panic!("Invalid digit"),
        }
    }
}

// Define MnistImage with Clone trait
#[derive(Clone, Debug)] // Derive Clone for MnistImage
pub struct MnistImage {
    pub data: MnistImageData,
    pub label: MnistDigit,
}

// Type alias for MnistImageData
pub type MnistImageData = Vec<Vec<u8>>;

// Enum for dataset types
pub enum MnistDataset {
    Train,
    Validate,
}

// Function to read image data from buffer
fn read_mnist_image_file(buffer: Vec<u8>) -> Vec<MnistImageData> {
    assert!(buffer[0] == 0 && buffer[1] == 0 && buffer[2] == 8 && buffer[3] == 3); // Magic number

    let num_images = u32::from_be_bytes(buffer[4..8].try_into().unwrap()) as usize;
    let num_rows = u32::from_be_bytes(buffer[8..12].try_into().unwrap()) as usize;
    let num_cols = u32::from_be_bytes(buffer[12..16].try_into().unwrap()) as usize;

    assert!(num_rows == 28 && num_cols == 28); // Image dimensions

    let mut offset = 16;
    let mut mnist_images: Vec<MnistImageData> = Vec::new();

    for _ in 0..num_images {
        let mut image_data: Vec<Vec<u8>> = Vec::new();
        for _ in 0..28 {
            let mut row = Vec::new();
            for _ in 0..28 {
                row.push(buffer[offset]);
                offset += 1;
            }
            image_data.push(row);
        }
        mnist_images.push(image_data);
    }

    mnist_images
}

// Function to read label data from buffer
fn read_mnist_label_file(buffer: Vec<u8>) -> Vec<MnistDigit> {
    assert!(buffer[0] == 0 && buffer[1] == 0 && buffer[2] == 8 && buffer[3] == 1); // Magic number

    let num_labels = u32::from_be_bytes(buffer[4..8].try_into().unwrap()) as usize;

    let mut offset = 8;
    let mut mnist_labels: Vec<MnistDigit> = Vec::new();

    for _ in 0..num_labels {
        let label = MnistDigit::from_usize(buffer[offset] as usize);
        mnist_labels.push(label);
        offset += 1;
    }

    mnist_labels
}

// Function to load the MNIST dataset
pub fn load_mnist_dataset(dataset: MnistDataset) -> Vec<MnistImage> {
    let (image_buffer, label_buffer) = match dataset {
        MnistDataset::Train => (
            include_bytes!("../data/train-images.idx3-ubyte").to_vec(),
            include_bytes!("../data/train-labels.idx1-ubyte").to_vec(),
        ),
        MnistDataset::Validate => (
            include_bytes!("../data/t10k-images.idx3-ubyte").to_vec(),
            include_bytes!("../data/t10k-labels.idx1-ubyte").to_vec(),
        ),
    };

    let image_data = read_mnist_image_file(image_buffer);
    let label_data = read_mnist_label_file(label_buffer);

    assert!(image_data.len() == label_data.len()); // Check if Image and Label data have the same number of elements

    // Combine Image and Label data
    image_data
        .iter()
        .zip(label_data.iter())
        .map(|(image, label)| MnistImage {
            data: image.clone(),
            label: *label,
        })
        .collect()
}
