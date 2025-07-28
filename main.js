const datasetSelect = document.getElementById("datasetSelect");
const modelSelect = document.getElementById("modelSelect");
const batchSizeInput = document.getElementById("batchSizeInput");
const useSeedCheckbox = document.getElementById("useSeedCheckbox");
const seedInput = document.getElementById("seedInput");
const generateButton = document.getElementById("generateButton");
const grid = document.getElementById("grid");
const classSelect = document.getElementById("classSelect");

const FASHION_MNIST_CLASSES = [
  "random", "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
];

const CIFAR10_CLASSES = [
  "random", "Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"
];

generateButton.addEventListener("click", async () => {
    // Clear previous images displayed in the grid and display a new one
    grid.innerHTML = "";
    await generateImages();
});

useSeedCheckbox.addEventListener("change", () => {
    seedInput.disabled = !useSeedCheckbox.checked; // Disable seed input if checkbox is unchecked
    if (!useSeedCheckbox.checked) seedInput.value = ""; // Clear seed input if checkbox is unchecked
});

function updateClasses() {
    const model = modelSelect.value;
    if (model === "cgan") {
        const dataset = datasetSelect.value; // Now updated
        classSelect.disabled = false;
        classSelect.innerHTML = "";

        const classList = dataset === "fashion_mnist"
            ? FASHION_MNIST_CLASSES
            : dataset === "cifar10"
                ? CIFAR10_CLASSES
                : [];

        classList.forEach((cls) => {
            const opt = document.createElement("option");
            opt.value = cls;
            opt.textContent = cls === "random" ? "Random class" : cls.charAt(0).toUpperCase() + cls.slice(1);
            classSelect.appendChild(opt);
        });
    } else {
        classSelect.disabled = true;
        classSelect.innerHTML = "";
    }
}

modelSelect.addEventListener("change", () => {
    const model = modelSelect.value;
    const datasetOptions = {
        vae: ["mnist", "fashion_mnist"],
        gan: ["mnist", "fashion_mnist"],
        cgan: ["fashion_mnist", "cifar10"]
    };

    const allowed = datasetOptions[model] || [];

    // Clear and rebuild datasetSelect options
    datasetSelect.innerHTML = "";
    allowed.forEach(ds => {
        const opt = document.createElement("option");
        opt.value = ds;
        opt.textContent = ds.replace("_", " ").toUpperCase();
        datasetSelect.appendChild(opt);
    });

    // Select the first option by default to ensure it's consistent
    datasetSelect.selectedIndex = 0;

    updateClasses();
});

datasetSelect.addEventListener("change", () => {
    updateClasses();
});

// Asynchronous function to generate images
async function generateImages() {
    const model = modelSelect.value; // Get selected model option
    if (model === "vae")
        await generateImagesVAE();
    else if (model === "gan")
        await generateImagesGAN();
    else if (model === "cgan")
        await generateImagesCGAN();
}

// Function to generate images for VAE
async function generateImagesVAE() {
    const Z_DIM = 100;
    const X_DIM = 56;
    const dataset = datasetSelect.value;
    const batchSize = Math.min(parseInt(batchSizeInput.value, 10) || 1, 45);
    let seed = parseInt(seedInput.value, 10);
    if (!useSeedCheckbox.checked || isNaN(seed))
        seed = Math.floor(Math.random() * 1e9);

    try {
        let modelPath = "Models/";
        if (dataset === "mnist")
            modelPath += "VAE_MNIST.onnx";
        else if (dataset === "fashion_mnist")
            modelPath += "VAE_Fashion_MNIST.onnx";

        const session = await ort.InferenceSession.create(modelPath);

        const latentData = generateLatentVector(seed, batchSize, Z_DIM);
        const inputTensor = new ort.Tensor("float32", latentData, [batchSize, Z_DIM]);

        const results = await session.run({ latent_vector: inputTensor });
        const generatedImages = results.generated_image.data;

        const imageSize = X_DIM * X_DIM;
        for (let i = 0; i < batchSize; i++) {
            const imageStart = i * imageSize;
            const imageEnd = imageStart + imageSize;
            const imageSlice = generatedImages.slice(imageStart, imageEnd);
            renderImage(imageSlice, X_DIM, X_DIM, 1, 0, 1);
        }
    } catch (error) {
        console.error("Error during generation:", error);
    }
}

// Function to generate images for GAN
async function generateImagesGAN() {
    const Z_DIM = 100; // Latent vector size
    const X_DIM = 56 // Image size
    const dataset = datasetSelect.value; // Get selected dataset option
    const batchSize = Math.min(parseInt(batchSizeInput.value, 10) || 1, 45); // Limit batch size to 45
    let seed = parseInt(seedInput.value, 10); // Get the seed value from input
    if (!useSeedCheckbox.checked || isNaN(seed))
        seed = Math.floor(Math.random() * 1e9);

    try {
        // Generate latent vector for input to model
        const latentVector = generateLatentVector(seed, batchSize, Z_DIM);

        // Set the path to the model based on the selected dataset
        let modelPath = "Models/";
        if (dataset === "mnist")
            modelPath += "GAN_MNIST.onnx"; // Model for MNIST dataset
        else if (dataset === "fashion_mnist")
            modelPath += "GAN_Fashion_MNIST.onnx"; // Model for Fashion MNIST dataset

        // Create ONNX InferenceSession for the selected model
        const session = await ort.InferenceSession.create(modelPath);

        // Create tensor from latent vector for input to the model
        const inputTensor = new ort.Tensor("float32", latentVector, [batchSize, Z_DIM, 1, 1]);

        // Run the model with the latent vector as input
        const results = await session.run({ latent_vector: inputTensor });
        const generatedImages = results.generated_image.data;

        // Render each generated image on the page
        const imageSize = X_DIM * X_DIM;
        for (let i = 0; i < batchSize; i++) {
            const imageStart = i * imageSize;
            const imageEnd = imageStart + imageSize;
            const imageSlice = generatedImages.slice(imageStart, imageEnd);
            renderImage(imageSlice, X_DIM, X_DIM, 1, -1, 1);
        }
    } catch (error) {
        console.error("Error during generation:", error); // Log errors if any occur
    }
}


// Function to generate images for cGAN (with class guidance)
async function generateImagesCGAN() {
    const Z_DIM = 100;
    const X_DIM = 56;
    const NUM_CLASSES = 10;

    const dataset  = datasetSelect.value;
    const batchSize = Math.min(parseInt(batchSizeInput.value, 10) || 1, 45);

    const NUM_CHANNELS = getChannelNum(dataset)
    const modelPath = getModelPath("cgan", dataset);
    
    // pick or randomize seed
    let seed = parseInt(seedInput.value, 10);
    if (!useSeedCheckbox.checked || isNaN(seed))
        seed = Math.floor(Math.random() * 1e9);

    // Determine class: either random or fixed
    let classLabels;
    if (classSelect.disabled || classSelect.value === "random") {
        classLabels = generateRandomLabels(seed, batchSize, NUM_CLASSES);
    } else {
        const classList = dataset === "fashion_mnist" ? FASHION_MNIST_CLASSES : CIFAR10_CLASSES;
        const clsIndex = classList.indexOf(classSelect.value) - 1;
        classLabels = Array(batchSize).fill(clsIndex);
    }

    try {
        // Generate latent vectors
        const latentData = generateLatentVector(seed, batchSize, Z_DIM);
        const latentTensor = new ort.Tensor(
            'float32',
            latentData,
            [batchSize, Z_DIM, 1, 1]
        );

        // pack into BigInt64Array
        const labels = new BigInt64Array(batchSize);
        for (let i = 0; i < batchSize; i++) {
            labels[i] = BigInt(classLabels[i]);
        }
        const labelTensor = new ort.Tensor("int64", labels, [batchSize]);

        // Run inference
        const session = await ort.InferenceSession.create(modelPath);
        const results = await session.run({
            latent_vector: latentTensor,
            class_label: labelTensor
        });
        const imgData = results.generated_image.data;

        // Render: each image has CHANNEL planes, row‑major
        const imgSize = X_DIM * X_DIM;
        for (let n = 0; n < batchSize; n++) {
            const base = n * NUM_CHANNELS * imgSize;
            const singleImage = imgData.slice(base, base + NUM_CHANNELS * imgSize);
            renderImage(singleImage, X_DIM, X_DIM, NUM_CHANNELS);
        }

    } catch (error) {
        console.error("Error during cGAN generation:", error);
    }
}


// Seedable PRNG: mulberry32
function splitmix32(a) {
    return function() {
        a |= 0;
        a = a + 0x9e3779b9 | 0;
        let t = a ^ a >>> 16;
        t = Math.imul(t, 0x21f0aaad);
        t = t ^ t >>> 15;
        t = Math.imul(t, 0x735a2d97);
        return ((t = t ^ t >>> 15) >>> 0) / 4294967296;
    }
}

// Gaussian noise using Marsaglia polar method
function generateLatentVector(seed, batchSize, dim) {
    if (isNaN(seed))
        seed = Math.round(Math.random() * 1e6)
    const rand = splitmix32(seed);
    const totalSize = batchSize * dim;
    const z = new Float32Array(totalSize);

    let i = 0;
    while (i < totalSize) {
        const u = 1 - rand(); // Converting [0,1) to (0,1]
        const v = rand();
        z[i++] = Math.sqrt( -2.0 * Math.log(u)) * Math.cos( 2.0 * Math.PI * v);
    }
    return z;
}

function generateRandomLabels(seed, batchSize, n_classes) {
    if (isNaN(seed))
        seed = Math.round(Math.random() * 1e6)
    const rand = splitmix32(seed);
    const z = new Float32Array(batchSize);

    let i = 0;
    while (i < batchSize) {
        const u = 1 - rand(); // Converting [0,1) to (0,1]
        const v = rand();
        const x = Math.sqrt( -2.0 * Math.log(u)) * Math.cos( 2.0 * Math.PI * v)
        z[i++] = Math.round(x * 1e9) % n_classes;
    }
    return z;
}

// Function to render the generated image on the page
function renderImage(imageData, x_dim, y_dim, channel, min = -1, max = 1) {
    const canvas = document.createElement("canvas");
    canvas.width = x_dim;
    canvas.height = y_dim;
    const ctx = canvas.getContext("2d");
    const imageDataObj = ctx.createImageData(x_dim, y_dim);

    const imgSize = x_dim * y_dim;
    const scale = (val) => ((val - min) / (max - min)) * 255;

    for (let i = 0; i < imgSize; i++) {
        const idx = i * 4;
        let r, g, b;

        if (channel === 1) {
            const val = scale(imageData[i]);
            r = g = b = val;
        } else if (channel === 3) {
            r = scale(imageData[i]);
            g = scale(imageData[i + imgSize]);
            b = scale(imageData[i + 2 * imgSize]);
        } else {
            throw new Error(`Unsupported channel count: ${channel}`);
        }

        imageDataObj.data[idx]     = r;
        imageDataObj.data[idx + 1] = g;
        imageDataObj.data[idx + 2] = b;
        imageDataObj.data[idx + 3] = 255;
    }

    ctx.putImageData(imageDataObj, 0, 0);
    grid.appendChild(canvas);
}

function getModelPath(model, dataset) {
    const map = {
        vae: {
            mnist: "VAE_MNIST.onnx",
            fashion_mnist: "VAE_Fashion_MNIST.onnx"
        },
        gan: {
            mnist: "GAN_MNIST.onnx",
            fashion_mnist: "GAN_Fashion_MNIST.onnx"
        },
        cgan: {
            fashion_mnist: "cGAN_Fashion_MNIST.onnx",
            cifar10: "cGAN_CIFAR10.onnx"
        }
    };
    const filename = map[model]?.[dataset];
    if (!filename) throw new Error(`Model path not found for ${model} + ${dataset}`);
    return "Models/" + filename;
}

function getChannelNum(dataset) {
    const map = {
        mnist: 1,
        fashion_mnist: 1,
        cifar10: 3
    };
    return map[dataset]
}

modelSelect.dispatchEvent(new Event("change"));
