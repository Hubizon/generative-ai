const datasetSelect = document.getElementById("datasetSelect");
const modelSelect = document.getElementById("modelSelect");
const batchSizeInput = document.getElementById("batchSizeInput");
const useSeedCheckbox = document.getElementById("useSeedCheckbox");
const seedInput = document.getElementById("seedInput");
const generateButton = document.getElementById("generateButton");
const grid = document.getElementById("grid");

generateButton.addEventListener("click", async () => {
    // Clear previous images displayed in the grid and display a new one
    grid.innerHTML = "";
    await generateImages();
});

useSeedCheckbox.addEventListener("change", () => {
    seedInput.disabled = !useSeedCheckbox.checked; // Disable seed input if checkbox is unchecked
    if (!useSeedCheckbox.checked) seedInput.value = ""; // Clear seed input if checkbox is unchecked
});

modelSelect.addEventListener("change", () => {
    const model = modelSelect.value;
    const datasetOptions = {
        gan: ["mnist", "fashion_mnist"],
        vae: ["mnist"]
    };

    const allowed = datasetOptions[model] || [];

    datasetSelect.innerHTML = "";

    allowed.forEach(ds => {
        const opt = document.createElement("option");
        opt.value = ds;
        opt.textContent = ds.replace("_", " ").toUpperCase();
        datasetSelect.appendChild(opt);
    });
});

// Asynchronous function to generate images
async function generateImages() {
    const model = modelSelect.value; // Get selected model option
    if (model === "gan")
        await generateImagesGAN();
    else if (model === "vae")
        await generateImagesVAE();
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
            renderImage(imageSlice, X_DIM, X_DIM, -1, 1);
        }
    } catch (error) {
        console.error("Error during generation:", error); // Log errors if any occur
    }
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
            renderImage(imageSlice, X_DIM, X_DIM, 0, 1);
        }
    } catch (error) {
        console.error("Error during generation:", error);
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

// Function to render the generated image on the page
function renderImage(imageData, x_dim, y_dim, min=-1, max=1) {
    const canvas = document.createElement("canvas");
    canvas.width = x_dim;
    canvas.height = y_dim;
    const ctx = canvas.getContext("2d");
    const imageDataObj = ctx.createImageData(x_dim, y_dim);

    // Assume range is either [-1, 1] or [0, 1]
    const scale = (val) => {
        return ((val - min) / (max - min)) * 255;
    };

    for (let i = 0; i < imageData.length; i++) {
        const value = scale(imageData[i]);
        imageDataObj.data[i * 4] = value;       // Red
        imageDataObj.data[i * 4 + 1] = value;   // Green
        imageDataObj.data[i * 4 + 2] = value;   // Blue
        imageDataObj.data[i * 4 + 3] = 255;     // Alpha
    }

    ctx.putImageData(imageDataObj, 0, 0);
    grid.appendChild(canvas);
}

modelSelect.dispatchEvent(new Event("change"));
