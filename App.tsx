
import React, { useState, useCallback, DragEvent, useRef, useEffect } from 'react';
import { GoogleGenAI, Type } from '@google/genai';

// --- TYPE DEFINITIONS ---
type Theme = 'light' | 'dark';

interface CompressionStats {
  originalSize: number;
  compressedSize: number;
  reductionPercentage: number;
}

interface ProcessedImage {
  dataUrl: string;
  name: string;
}

interface BoundingBox {
  box: [number, number, number, number]; // [x_min, y_min, x_max, y_max] normalized
  label: string;
}

interface AnalysisResult {
  plantIdentification: {
    species: string;
    notes: string;
  };
  diagnosis: {
    isConfident: boolean;
    confidenceScore: number;
    diseaseName: string;
    scientificName: string;
    description: string;
    differentialDiagnosis: {
      diseaseName: string;
      reasoning: string;
    }[];
  };
  treatmentOptions: {
    organic: string[];
    chemical: string[];
  };
  preventiveMeasures: string[];
  disclaimer: string;
}

// --- GEMINI API SCHEMAS ---

const analysisResponseSchema = {
  type: Type.OBJECT,
  properties: {
    plantIdentification: {
      type: Type.OBJECT,
      properties: {
        species: { type: Type.STRING, description: "Scientific name (and common name) of the plant, e.g., 'Solanum lycopersicum (Tomato)'." },
        notes: { type: Type.STRING, description: "Brief observation about the plant's cultivar or general state, if relevant." },
      },
      required: ['species', 'notes']
    },
    diagnosis: {
      type: Type.OBJECT,
      properties: {
        isConfident: { type: Type.BOOLEAN, description: "True if confidence is high (>= 0.80), false otherwise." },
        confidenceScore: { type: Type.NUMBER, description: "A score from 0.0 to 1.0 indicating confidence in the diagnosis." },
        diseaseName: { type: Type.STRING, description: "Common name of the identified disease. If healthy, state 'Healthy'." },
        scientificName: { type: Type.STRING, description: "Scientific name of the pathogen, or 'N/A' if healthy or not applicable." },
        description: { type: Type.STRING, description: "Detailed description of the visual symptoms observed on the leaf that support the diagnosis." },
        differentialDiagnosis: {
          type: Type.ARRAY,
          items: {
            type: Type.OBJECT,
            properties: {
              diseaseName: { type: Type.STRING, description: "Name of a similar disease that was considered but ruled out." },
              reasoning: { type: Type.STRING, description: "The specific reason why this alternative diagnosis was excluded based on visual evidence." },
            },
            required: ['diseaseName', 'reasoning']
          }
        }
      },
      required: ['isConfident', 'confidenceScore', 'diseaseName', 'scientificName', 'description', 'differentialDiagnosis']
    },
    treatmentOptions: {
      type: Type.OBJECT,
      properties: {
        organic: { type: Type.ARRAY, items: { type: Type.STRING }, description: "List of organic treatment methods." },
        chemical: { type: Type.ARRAY, items: { type: Type.STRING }, description: "List of chemical treatment methods." },
      },
      required: ['organic', 'chemical']
    },
    preventiveMeasures: {
      type: Type.ARRAY,
      items: { type: Type.STRING },
      description: "A list of measures to prevent future occurrences of this disease."
    },
    disclaimer: { type: Type.STRING, description: "A standard disclaimer stating this is an AI-generated analysis and professional advice should be sought." },
  },
  required: ['plantIdentification', 'diagnosis', 'treatmentOptions', 'preventiveMeasures', 'disclaimer']
};

const visualExplanationSchema = {
    type: Type.ARRAY,
    items: {
        type: Type.OBJECT,
        properties: {
            box: {
                type: Type.ARRAY,
                items: { type: Type.NUMBER },
                description: "A list of 4 numbers representing the normalized bounding box coordinates [x_min, y_min, x_max, y_max]."
            },
            label: {
                type: Type.STRING,
                description: "A very brief label for the bounding box, e.g., 'Necrotic Lesion' or 'Fungal Spores'."
            }
        },
        required: ['box', 'label']
    }
};


// --- GEMINI API SERVICE ---

/**
 * Analyzes an image with Gemini to diagnose plant disease.
 * @returns A promise that resolves with the analysis object.
 */
const analyzeImageWithGemini = async (imageDataUrl: string, mimeType: 'image/jpeg' | 'image/png'): Promise<AnalysisResult> => {
    const ai = new GoogleGenAI({ apiKey: process.env.API_KEY as string });
    
    const base64Data = imageDataUrl.split(',')[1];
    const imagePart = {
        inlineData: {
            mimeType: mimeType,
            data: base64Data,
        },
    };
    const textPart = {
        text: "Analyze this plant leaf image and provide a detailed diagnosis. Identify the plant species and the disease, if any. Describe the symptoms, provide treatment options, and suggest preventive measures. Also, include a differential diagnosis section explaining why other similar diseases were ruled out. Your response must be in the specified JSON format."
    };

    try {
        const response = await ai.models.generateContent({
            model: 'gemini-2.5-pro',
            contents: { parts: [imagePart, textPart] },
            config: {
                systemInstruction: "You are a world-renowned botanist and plant pathologist AI. Your task is to accurately diagnose plant diseases from leaf images and provide comprehensive, actionable advice to gardeners and farmers. You must strictly adhere to the provided JSON schema for your response.",
                temperature: 0.1,
                thinkingConfig: { thinkingBudget: 16384 },
                responseMimeType: "application/json",
                responseSchema: analysisResponseSchema,
            }
        });

        // Use response.text and parse it, as it's guaranteed by the schema to be JSON.
        return JSON.parse(response.text);
    } catch (e) {
        console.error("Error analyzing image with Gemini:", e);
        if (e instanceof Error) {
           throw new Error(`Gemini API Error: ${e.message}`);
        }
        throw new Error("An unknown error occurred during Gemini API call.");
    }
};

/**
 * Gets visual explanations (bounding boxes) from Gemini.
 * @returns A promise that resolves with an array of bounding boxes.
 */
const getVisualExplanation = async (imageDataUrl: string, mimeType: 'image/jpeg' | 'image/png', diagnosisText: string): Promise<BoundingBox[]> => {
    const ai = new GoogleGenAI({ apiKey: process.env.API_KEY as string });

    const base64Data = imageDataUrl.split(',')[1];
    const imagePart = {
      inlineData: {
            mimeType: mimeType,
            data: base64Data,
        },
    };
    const textPart = { 
        text: `Based on the diagnosis that the leaf shows symptoms of "${diagnosisText}", identify the most prominent visual indicators of the disease in the image. For each key symptom (like lesions, spots, or mildew), provide a normalized bounding box. Your response must be an array of objects in the specified JSON format.`
    };

    try {
        const response = await ai.models.generateContent({
            model: 'gemini-2.5-pro',
            contents: { parts: [imagePart, textPart] },
            config: {
                temperature: 0.0,
                responseMimeType: "application/json",
                responseSchema: visualExplanationSchema,
            }
        });

        const result = JSON.parse(response.text);
        
        // Basic validation of the returned data structure
        if (Array.isArray(result)) {
            return result.filter(item => 
                item.box && Array.isArray(item.box) && item.box.length === 4 && typeof item.label === 'string'
            );
        }
        console.warn("Visual explanation response was not an array:", result);
        return [];

    } catch (e) {
        console.error("Failed to parse or get visual explanation from Gemini:", e);
        if (e instanceof Error) {
            console.error("Gemini Response Text:", (e as any).response?.text);
        }
        return []; // Return empty array on failure instead of throwing, as this is non-critical.
    }
};


// --- UTILITY & SERVICE FUNCTIONS ---

interface VQResult {
  finalDataUrl: string;
  stats: CompressionStats;
  mimeType: 'image/jpeg' | 'image/png';
  compressionApplied: boolean;
}

/**
 * Implements a performant, conditional client-side Vector Quantization (VQ)
 * using pixel-based color quantization. It compresses the image but only returns
 * the compressed version if it's smaller than the original file.
 * @param file The image file to be compressed.
 * @returns A promise resolving with the final image data, stats, and a flag indicating if compression was applied.
 */
const compressWithVectorQuantization = (file: File): Promise<VQResult> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = (e) => {
      const originalDataUrl = e.target?.result as string;
      if (!originalDataUrl) {
          return reject(new Error("Could not read file data."));
      }
      const img = new Image();
      img.src = originalDataUrl;
      img.onload = () => {
        // --- VQ Parameters ---
        const K = 64; // Number of colors in the final palette
        const MAX_ITERATIONS = 10;
        const PIXEL_SAMPLE_SIZE = 40000; // Max pixels to train K-Means on
        const MAX_DIMENSION = 256;

        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d', { willReadFrequently: true });
        if (!ctx) return reject(new Error('Failed to get canvas context'));

        // 1. Pre-process: Resize image
        let { width, height } = img;
        if (width > height) {
          if (width > MAX_DIMENSION) {
            height = Math.round((height * MAX_DIMENSION) / width);
            width = MAX_DIMENSION;
          }
        } else {
          if (height > MAX_DIMENSION) {
            width = Math.round((width * MAX_DIMENSION) / height);
            height = MAX_DIMENSION;
          }
        }
        canvas.width = width;
        canvas.height = height;
        ctx.drawImage(img, 0, 0, width, height);
        const imageData = ctx.getImageData(0, 0, width, height);
        const { data: imgData, width: imgWidth, height: imgHeight } = imageData;

        // 2. Vectorization (Pixel-based)
        const pixels: number[][] = [];
        for (let i = 0; i < imgData.length; i += 4) {
            if (imgData[i + 3] > 128) { // Only consider visible pixels
                pixels.push([imgData[i], imgData[i + 1], imgData[i + 2]]);
            }
        }
        if (pixels.length === 0) return reject(new Error("Image appears to be empty or fully transparent."));


        // 3. Codebook Generation (K-Means)
        const trainingPixels = pixels.length > PIXEL_SAMPLE_SIZE
          ? [...pixels].sort(() => 0.5 - Math.random()).slice(0, PIXEL_SAMPLE_SIZE)
          : pixels;

        let codebook: number[][] = trainingPixels.slice(0, K).map(p => [...p]);
        const assignments = new Array(trainingPixels.length);
        const euclideanDistanceSq = (v1: number[], v2: number[]) => (v1[0] - v2[0]) ** 2 + (v1[1] - v2[1]) ** 2 + (v1[2] - v2[2]) ** 2;

        for (let iter = 0; iter < MAX_ITERATIONS; iter++) {
          for (let i = 0; i < trainingPixels.length; i++) {
            let minDistance = Infinity;
            let bestIndex = 0;
            for (let j = 0; j < codebook.length; j++) {
              const distance = euclideanDistanceSq(trainingPixels[i], codebook[j]);
              if (distance < minDistance) {
                minDistance = distance;
                bestIndex = j;
              }
            }
            assignments[i] = bestIndex;
          }

          const newCodebook = Array.from({ length: K }, () => [0, 0, 0]);
          const counts = new Array(K).fill(0);
          for (let i = 0; i < trainingPixels.length; i++) {
            const pixel = trainingPixels[i];
            const assignment = assignments[i];
            newCodebook[assignment][0] += pixel[0];
            newCodebook[assignment][1] += pixel[1];
            newCodebook[assignment][2] += pixel[2];
            counts[assignment]++;
          }

          let hasChanged = false;
          for (let i = 0; i < codebook.length; i++) {
            if (counts[i] > 0) {
              const newCentroid = newCodebook[i].map(val => Math.round(val / counts[i]));
              if (euclideanDistanceSq(newCentroid, codebook[i]) > 1e-4) {
                  hasChanged = true;
              }
              codebook[i] = newCentroid;
            } else { // Re-initialize empty clusters
              codebook[i] = trainingPixels[Math.floor(Math.random() * trainingPixels.length)];
            }
          }
          if (!hasChanged) break;
        }

        // 4. Reconstruction
        const newImageData = ctx.createImageData(imgWidth, imgHeight);
        const pixelMap = new Map<string, number[]>(); // Cache color lookups
        for (let i = 0; i < imgData.length; i += 4) {
            const r = imgData[i], g = imgData[i + 1], b = imgData[i + 2];
            const originalColorKey = `${r},${g},${b}`;
            let finalColor: number[];

            if (pixelMap.has(originalColorKey)) {
                finalColor = pixelMap.get(originalColorKey)!;
            } else {
                let bestIndex = 0;
                let minDistance = Infinity;
                for (let j = 0; j < codebook.length; j++) {
                    const distance = (r - codebook[j][0]) ** 2 + (g - codebook[j][1]) ** 2 + (b - codebook[j][2]) ** 2;
                    if (distance < minDistance) {
                        minDistance = distance;
                        bestIndex = j;
                    }
                }
                finalColor = codebook[bestIndex];
                pixelMap.set(originalColorKey, finalColor);
            }
            newImageData.data[i] = finalColor[0];
            newImageData.data[i + 1] = finalColor[1];
            newImageData.data[i + 2] = finalColor[2];
            newImageData.data[i + 3] = imgData[i + 3]; // Preserve alpha
        }
        ctx.putImageData(newImageData, 0, 0);

        // 5. Encode and decide whether to use the compressed version
        const originalSize = file.size;
        const vqCompressedDataUrl = canvas.toDataURL('image/png');
        const vqCompressedSize = atob(vqCompressedDataUrl.split(',')[1]).length;

        if (vqCompressedSize < originalSize) {
          resolve({
            finalDataUrl: vqCompressedDataUrl,
            stats: { 
              originalSize, 
              compressedSize: vqCompressedSize, 
              reductionPercentage: ((originalSize - vqCompressedSize) / originalSize) * 100 
            },
            mimeType: 'image/png',
            compressionApplied: true,
          });
        } else {
          resolve({
            finalDataUrl: originalDataUrl,
            stats: {
              originalSize,
              compressedSize: originalSize,
              reductionPercentage: 0,
            },
            mimeType: file.type as 'image/jpeg' | 'image/png',
            compressionApplied: false,
          });
        }
      };
      img.onerror = () => reject(new Error("Failed to load image for Vector Quantization. The file may be corrupt."));
    };
    reader.onerror = () => reject(new Error("Failed to read the selected file."));
  });
};

// --- UI HELPER COMPONENTS ---

const Header = () => (
  <header className="text-center pt-12">
    <div className="inline-flex items-center gap-3 bg-primary-100 dark:bg-primary-900 text-primary-700 dark:text-primary-200 py-2 px-4 rounded-full">
      <LeafIcon />
      <h1 className="text-2xl md:text-3xl font-bold">AI Leaf Disease Detector</h1>
    </div>
    <p className="mt-2 text-md text-gray-600 dark:text-gray-400 max-w-2xl mx-auto">
      Upload an image of a plant leaf to get an instant, expert diagnosis from our advanced AI.
    </p>
  </header>
);

const ThemeToggle: React.FC<{ theme: Theme; toggleTheme: () => void; }> = ({ theme, toggleTheme }) => (
    <button
      onClick={toggleTheme}
      className="absolute top-4 right-4 p-2 rounded-full text-gray-500 dark:text-gray-400 hover:bg-gray-200 dark:hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500 transition-colors"
      aria-label="Toggle theme"
    >
      {theme === 'dark' ? <SunIcon /> : <MoonIcon />}
    </button>
);

const FileUploader: React.FC<{ onFileSelect: (file: File) => void; disabled: boolean }> = ({ onFileSelect, disabled }) => {
  const [isDragging, setIsDragging] = useState(false);

  const handleDrag = (e: DragEvent<HTMLDivElement>, enter: boolean) => {
    e.preventDefault();
    e.stopPropagation();
    if (!disabled) setIsDragging(enter);
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    if (!disabled && e.dataTransfer.files && e.dataTransfer.files[0]) {
      onFileSelect(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) onFileSelect(e.target.files[0]);
  };

  const baseClasses = "relative block w-full rounded-lg border-2 border-dashed p-12 text-center transition-colors duration-300";
  const draggingClasses = "border-primary-500 bg-primary-50 dark:bg-primary-900/50";
  const defaultClasses = "border-gray-300 dark:border-gray-600 hover:border-primary-400 dark:hover:border-primary-500";
  const disabledClasses = "bg-gray-100 dark:bg-gray-800 cursor-not-allowed opacity-50";

  const getClassName = () => {
    if (disabled) return `${baseClasses} ${disabledClasses}`;
    if (isDragging) return `${baseClasses} ${draggingClasses}`;
    return `${baseClasses} ${defaultClasses}`;
  };

  return (
     <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-md mb-8">
      <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Upload Your Image</h3>
      <div
        className={getClassName()}
        onDragEnter={(e) => handleDrag(e, true)}
        onDragLeave={(e) => handleDrag(e, false)}
        onDragOver={(e) => e.preventDefault()}
        onDrop={handleDrop}
      >
        <UploadIcon />
        <span className="mt-2 block font-semibold text-gray-900 dark:text-gray-100">
          Drag & drop a file here, or click to select a file
        </span>
        <span className="mt-1 block text-sm text-gray-500 dark:text-gray-400">PNG, JPG up to 10MB</span>
        <input
          type="file"
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
          onChange={handleChange}
          accept="image/*"
          disabled={disabled}
        />
      </div>
    </div>
  );
};

const AnalysisReport: React.FC<{ report: AnalysisResult }> = ({ report }) => {
  const { plantIdentification, diagnosis, treatmentOptions, preventiveMeasures, disclaimer } = report;

  const ConfidenceIndicator: React.FC<{ score: number }> = ({ score }) => {
    const percentage = Math.round(score * 100);
    let colorClass = 'bg-green-500';
    if (percentage < 75) colorClass = 'bg-yellow-500';
    if (percentage < 50) colorClass = 'bg-red-500';

    return (
        <div className="flex items-center gap-3">
            <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-4">
                <div className={`${colorClass} h-4 rounded-full`} style={{ width: `${percentage}%` }}></div>
            </div>
            <span className="font-bold text-lg text-gray-800 dark:text-gray-200">{percentage}%</span>
        </div>
    );
  };
  
  const Section: React.FC<{ title: string; children: React.ReactNode }> = ({ title, children }) => (
    <div className="pt-4">
      <h4 className="text-xl font-bold text-gray-900 dark:text-white border-b border-gray-200 dark:border-gray-700 pb-2 mb-3">{title}</h4>
      <div className="space-y-2 text-gray-700 dark:text-gray-300">{children}</div>
    </div>
  );

  const ListItem: React.FC<{ children: React.ReactNode }> = ({ children }) => (
    <li className="flex items-start">
        <svg className="h-5 w-5 mr-2 mt-1 text-primary-500 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20"><path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd"></path></svg>
        <span>{children}</span>
    </li>
  );

  return (
    <div className="space-y-4">
        {disclaimer && (
            <div className="p-4 bg-yellow-100 dark:bg-yellow-900/50 border-l-4 border-yellow-500 text-yellow-800 dark:text-yellow-200 rounded-r-lg" role="alert">
                <p className="font-semibold">Disclaimer</p>
                <p>{disclaimer}</p>
            </div>
        )}

        <Section title="Plant Identification">
            <p><strong className="font-semibold text-gray-800 dark:text-gray-200">Species:</strong> {plantIdentification.species}</p>
            {plantIdentification.notes && <p><strong className="font-semibold text-gray-800 dark:text-gray-200">Notes:</strong> {plantIdentification.notes}</p>}
        </Section>
        
        <Section title="Diagnosis">
            <h5 className="text-lg font-semibold text-gray-800 dark:text-gray-200">
                {diagnosis.diseaseName}
                {diagnosis.scientificName && diagnosis.scientificName !== 'N/A' && <span className="text-sm italic text-gray-500 ml-2">({diagnosis.scientificName})</span>}
            </h5>
            <p>{diagnosis.description}</p>
            <div className="mt-4">
                <p className="font-semibold text-gray-800 dark:text-gray-200 mb-1">Confidence:</p>
                <ConfidenceIndicator score={diagnosis.confidenceScore} />
                {!diagnosis.isConfident && <p className="text-sm text-yellow-600 dark:text-yellow-400 mt-2">The AI has low confidence in this diagnosis. Please consult a human expert for confirmation.</p>}
            </div>
        </Section>

        {diagnosis.differentialDiagnosis && diagnosis.differentialDiagnosis.length > 0 && (
          <Section title="Differential Diagnosis">
            <p className="text-sm mb-3 text-gray-600 dark:text-gray-400">The AI also considered and ruled out the following similar conditions to increase diagnostic accuracy:</p>
            <div className="space-y-3">
              {diagnosis.differentialDiagnosis.map((dd, index) => (
                <div key={index} className="p-3 bg-gray-100 dark:bg-gray-700/50 rounded-lg border border-gray-200 dark:border-gray-600">
                  <h6 className="font-semibold text-gray-800 dark:text-gray-200">{dd.diseaseName}</h6>
                  <p className="text-sm"><strong className="text-gray-600 dark:text-gray-300">Reasoning for Exclusion:</strong> {dd.reasoning}</p>
                </div>
              ))}
            </div>
          </Section>
        )}
        
        { (treatmentOptions.organic.length > 0 || treatmentOptions.chemical.length > 0) &&
          <Section title="Treatment Options">
              {treatmentOptions.organic.length > 0 && (
                  <>
                      <h5 className="text-md font-semibold text-gray-800 dark:text-gray-200 pt-2">Organic</h5>
                      <ul className="space-y-1 list-inside">
                          {treatmentOptions.organic.map((item, i) => <ListItem key={`org-${i}`}>{item}</ListItem>)}
                      </ul>
                  </>
              )}
              {treatmentOptions.chemical.length > 0 && (
                  <>
                      <h5 className="text-md font-semibold text-gray-800 dark:text-gray-200 pt-2">Chemical</h5>
                      <ul className="space-y-1 list-inside">
                          {treatmentOptions.chemical.map((item, i) => <ListItem key={`chem-${i}`}>{item}</ListItem>)}
                      </ul>
                  </>
              )}
          </Section>
        }

        { preventiveMeasures.length > 0 &&
          <Section title="Preventive Measures">
              <ul className="space-y-1 list-inside">
                  {preventiveMeasures.map((item, i) => <ListItem key={`prev-${i}`}>{item}</ListItem>)}
              </ul>
          </Section>
        }
    </div>
  );
};

const ResultsDisplay: React.FC<{
  originalImage: ProcessedImage;
  processedImage: ProcessedImage;
  stats: CompressionStats;
  result: AnalysisResult;
  compressionApplied: boolean;
  visualExplanation: BoundingBox[] | null;
  isLoadingExplanation: boolean;
}> = ({ originalImage, processedImage, stats, result, compressionApplied, visualExplanation, isLoadingExplanation }) => {
  const formatBytes = (bytes: number, decimals = 2) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
  };

  const handleDownload = () => {
    if (!processedImage) return;
    const link = document.createElement('a');
    link.href = processedImage.dataUrl;
    link.download = processedImage.name;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };
  
  return (
    <div className="space-y-8">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 opacity-0 animate-fade-in-up" style={{ animationDelay: '100ms' }}>
        <ImageCard title="Original Image" image={originalImage} />
        <ImageCard 
            title={compressionApplied ? "VQ Compressed" : "Original (Used for AI)"} 
            image={processedImage} 
            onDownload={compressionApplied ? handleDownload : undefined}
        />
        <ImageWithOverlays 
            title="AI Focus Map" 
            image={processedImage} 
            boxes={visualExplanation} 
            isLoading={isLoadingExplanation}
        />
      </div>

      <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-md opacity-0 animate-fade-in-up" style={{ animationDelay: '200ms' }}>
          <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
            <StatsIcon /> Compression Analysis
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
            <StatItem label="Original Size" value={formatBytes(stats.originalSize)} />
            <StatItem label="Final Size for AI" value={formatBytes(stats.compressedSize)} />
            <StatItem label="Size Reduction" value={`${stats.reductionPercentage.toFixed(2)}%`} className={stats.reductionPercentage > 0 ? "text-primary-600 dark:text-primary-400 font-bold" : ""} />
            <StatItem label="Method" value={compressionApplied ? 'Vector Quantization' : 'None (VQ not smaller)'} />
          </div>
      </div>
      
      <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-md opacity-0 animate-fade-in-up" style={{ animationDelay: '300ms' }}>
        <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
          <ReportIcon /> AI Analysis Report
        </h3>
        {result ? <AnalysisReport report={result} /> : <p className="text-center text-gray-600 dark:text-gray-400">Could not analyze the image.</p>}
      </div>

      <div className="opacity-0 animate-fade-in-up" style={{ animationDelay: '400ms' }}>
        <CompressionExplanation />
      </div>
    </div>
  );
};

const ImageCard: React.FC<{ title: string; image: ProcessedImage; onDownload?: () => void; }> = ({ title, image, onDownload }) => (
    <div className="bg-white dark:bg-gray-800 p-4 rounded-lg shadow-md transform transition-all duration-300 hover:scale-105 hover:shadow-xl">
        <div className="flex justify-between items-center mb-2">
            <h4 className="text-lg font-semibold text-gray-800 dark:text-white">{title}</h4>
            {onDownload && (
                <button onClick={onDownload} className="p-1.5 rounded-full text-gray-500 dark:text-gray-400 hover:bg-gray-200 dark:hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500 transition-colors" aria-label="Download image">
                    <DownloadIcon />
                </button>
            )}
        </div>
        <img src={image.dataUrl} alt={title} className="w-full h-auto object-contain rounded-md max-h-80" />
        <p className="text-xs text-gray-500 dark:text-gray-400 mt-2 truncate">{image.name}</p>
    </div>
);

const ImageWithOverlays: React.FC<{ title: string; image: ProcessedImage; boxes: BoundingBox[] | null; isLoading: boolean; }> = ({ title, image, boxes, isLoading }) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);

    useEffect(() => {
        if (!image || !canvasRef.current) return;
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;
        
        const img = new Image();
        img.src = image.dataUrl;
        img.onload = () => {
            canvas.width = img.width;
            canvas.height = img.height;
            ctx.drawImage(img, 0, 0);

            if (boxes && boxes.length > 0) {
                let startTime: number | null = null;
                const duration = 500; // 500ms fade-in

                const animate = (timestamp: number) => {
                    if (!startTime) startTime = timestamp;
                    const progress = timestamp - startTime;
                    const alpha = Math.min(progress / duration, 1);

                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    ctx.drawImage(img, 0, 0);

                    boxes.forEach(({ box }) => {
                        const [x_min, y_min, x_max, y_max] = box;
                        const x = x_min * img.width;
                        const y = y_min * img.height;
                        const w = (x_max - x_min) * img.width;
                        const h = (y_max - y_min) * img.height;
                        
                        ctx.strokeStyle = `rgba(239, 68, 68, ${alpha})`; // red-500
                        ctx.lineWidth = 3;
                        ctx.fillStyle = `rgba(239, 68, 68, ${alpha * 0.2})`; // red-500 with 20% opacity
                        
                        ctx.strokeRect(x, y, w, h);
                        ctx.fillRect(x, y, w, h);
                    });

                    if (progress < duration) {
                        requestAnimationFrame(animate);
                    }
                };
                requestAnimationFrame(animate);
            }
        };
        img.onerror = () => console.error("Failed to load image for canvas.");
    }, [image, boxes]);


    return (
        <div className="bg-white dark:bg-gray-800 p-4 rounded-lg shadow-md relative transform transition-all duration-300 hover:scale-105 hover:shadow-xl">
            <h4 className="text-lg font-semibold text-gray-800 dark:text-white mb-2">{title}</h4>
            <div className="relative w-full h-80 flex items-center justify-center">
                 {isLoading && (
                    <div className="absolute inset-0 flex flex-col items-center justify-center bg-gray-100/50 dark:bg-gray-800/50 rounded-md">
                        <svg aria-hidden="true" className="w-8 h-8 text-gray-200 animate-spin dark:text-gray-600 fill-primary-600" viewBox="0 0 100 101" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M100 50.5908C100 78.2051 77.6142 100.591 50 100.591C22.3858 100.591 0 78.2051 0 50.5908C0 22.9766 22.3858 0.59082 50 0.59082C77.6142 0.59082 100 22.9766 100 50.5908ZM9.08144 50.5908C9.08144 73.1895 27.4013 91.5094 50 91.5094C72.5987 91.5094 90.9186 73.1895 90.9186 50.5908C90.9186 27.9921 72.5987 9.67226 50 9.67226C27.4013 9.67226 9.08144 27.9921 9.08144 50.5908Z" fill="currentColor"/><path d="M93.9676 39.0409C96.393 38.4038 97.8624 35.9116 97.0079 33.5539C95.2932 28.8227 92.871 24.3692 89.8167 20.348C85.8452 15.1192 80.8826 10.7238 75.2124 7.41289C69.5422 4.10194 63.2754 1.94025 56.7698 1.05124C51.7666 0.367541 46.6976 0.446843 41.7345 1.27873C39.2613 1.69328 37.813 4.19778 38.4501 6.62326C39.0873 9.04874 41.5694 10.4717 44.0505 10.1071C47.8511 9.54855 51.7191 9.52689 55.5402 10.0492C60.8642 10.7766 65.9928 12.5457 70.6331 15.2552C75.2735 17.9648 79.3347 21.5619 82.5849 25.841C84.9175 28.9121 86.7997 32.2913 88.1811 35.8758C89.083 38.2158 91.5421 39.6781 93.9676 39.0409Z" fill="currentFill"/></svg>
                        <p className="mt-2 text-sm text-gray-600 dark:text-gray-400">Generating...</p>
                    </div>
                )}
                <canvas ref={canvasRef} className="max-w-full max-h-full object-contain rounded-md" />
            </div>
            <p className="text-xs text-gray-500 dark:text-gray-400 mt-2 truncate">{image.name}</p>
        </div>
    );
};

const StatItem: React.FC<{ label: string; value: string; className?: string }> = ({ label, value, className }) => (
    <div className="bg-gray-100 dark:bg-gray-700 p-3 rounded-md">
        <p className="text-sm text-gray-500 dark:text-gray-400">{label}</p>
        <p className={`text-lg font-semibold text-gray-900 dark:text-white ${className || ''}`}>{value}</p>
    </div>
);

const CompressionExplanation = () => (
  <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-md">
    <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-4">About the Compression</h3>
    <div className="space-y-3 text-gray-700 dark:text-gray-300">
      <p>
        This application uses an advanced color reduction technique called <strong>Vector Quantization (VQ)</strong> to prepare your image for the AI. It intelligently analyzes all the colors in your image and creates a new, optimized palette containing only the 64 most representative colors. Each pixel in the image is then replaced by its closest match from this limited palette.
      </p>
      <p>
        This process can significantly reduce the file size by simplifying color data, which helps the AI focus on structural patterns of disease—like shapes and textures—instead of subtle, often irrelevant, color gradients.
      </p>
      <p>
        <strong>When is compression applied?</strong> To ensure the best quality, compression is only used if the optimized image is actually smaller than your original file. If not, your original, untouched image is sent to the AI. This guarantees maximum efficiency without sacrificing critical detail.
      </p>
    </div>
  </div>
);


// --- SVG ICONS ---
const LeafIcon = () => (<svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>);
const UploadIcon = () => (<svg className="mx-auto h-12 w-12 text-gray-400" stroke="currentColor" fill="none" viewBox="0 0 48 48" aria-hidden="true"><path d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" /></svg>);
const StatsIcon = () => (<svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M13 10V3L4 14h7v7l9-11h-7z" /></svg>);
const ReportIcon = () => (<svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V7a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" /></svg>);
const MoonIcon = () => (<svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" /></svg>);
const SunIcon = () => (<svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" /></svg>);
const DownloadIcon = () => (<svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" /></svg>);


// --- MAIN APP COMPONENT ---
export default function App() {
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [isLoadingExplanation, setIsLoadingExplanation] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [originalImage, setOriginalImage] = useState<ProcessedImage | null>(null);
  const [processedImage, setProcessedImage] = useState<ProcessedImage | null>(null);
  const [compressionStats, setCompressionStats] = useState<CompressionStats | null>(null);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [visualExplanation, setVisualExplanation] = useState<BoundingBox[] | null>(null);
  const [compressionApplied, setCompressionApplied] = useState<boolean>(false);
  const [theme, setTheme] = useState<Theme>(() => {
    if (typeof window !== 'undefined' && localStorage.getItem('theme')) {
      return localStorage.getItem('theme') as Theme;
    }
    if (typeof window !== 'undefined' && window.matchMedia('(prefers-color-scheme: dark)').matches) {
      return 'dark';
    }
    return 'light';
  });

  useEffect(() => {
    const root = window.document.documentElement;
    root.classList.remove('light', 'dark');
    root.classList.add(theme);
    localStorage.setItem('theme', theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme(prevTheme => (prevTheme === 'dark' ? 'light' : 'dark'));
  };

  const resetState = () => {
    setError(null);
    setOriginalImage(null);
    setProcessedImage(null);
    setCompressionStats(null);
    setAnalysisResult(null);
    setVisualExplanation(null);
    setCompressionApplied(false);
  };

  const handleFileSelect = useCallback(async (file: File) => {
    if (!file.type.startsWith('image/')) {
      setError('Invalid file type. Please upload an image.');
      return;
    }

    setIsLoading(true);
    setIsLoadingExplanation(false);
    resetState();

    let dataUrlForAnalysis: string;
    let mimeTypeForAnalysis: 'image/jpeg' | 'image/png';

    try {
      // Set original image for display immediately
      const originalReader = new FileReader();
      originalReader.readAsDataURL(file);
      originalReader.onload = (e) => setOriginalImage({ dataUrl: e.target?.result as string, name: file.name });
      
      // Step 1: Process the image with VQ.
      const { finalDataUrl, stats, mimeType, compressionApplied: wasApplied } = await compressWithVectorQuantization(file);

      // Store results for the UI
      setProcessedImage({ dataUrl: finalDataUrl, name: wasApplied ? `vq_${file.name}` : file.name });
      setCompressionStats(stats);
      setCompressionApplied(wasApplied);
      
      // Keep a reference for the subsequent AI calls
      dataUrlForAnalysis = finalDataUrl;
      mimeTypeForAnalysis = mimeType;

      // Step 2: Send the FINAL image to get the main diagnosis
      const result = await analyzeImageWithGemini(dataUrlForAnalysis, mimeTypeForAnalysis);
      setAnalysisResult(result);
      
      setIsLoading(false);

      // Step 3: Asynchronously get the visual explanation.
      if (result.diagnosis.diseaseName !== 'Healthy') {
          setIsLoadingExplanation(true);
          try {
            const boxes = await getVisualExplanation(dataUrlForAnalysis, mimeTypeForAnalysis, result.diagnosis.diseaseName);
            setVisualExplanation(boxes);
          } catch (explanationError) {
             console.error("Failed to get visual explanation:", explanationError);
          } finally {
            setIsLoadingExplanation(false);
          }
      }

    } catch (err) {
      console.error("Processing failed:", err);
      const errorMessage = err instanceof Error ? err.message : 'An unknown error occurred.';
      setError(`An error occurred: ${errorMessage}. Please try again with a different image.`);
      setIsLoading(false);
      setIsLoadingExplanation(false);
    }
  }, []);

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 text-gray-800 dark:text-gray-200 font-sans p-4 sm:p-6 lg:p-8">
      <div className="container mx-auto max-w-7xl space-y-8 relative">
        <ThemeToggle theme={theme} toggleTheme={toggleTheme} />
        <Header />
        <main>
          <FileUploader onFileSelect={handleFileSelect} disabled={isLoading} />
          {error && <div className="mt-4 text-center text-red-500 bg-red-100 dark:bg-red-900/50 p-3 rounded-lg">{error}</div>}
          
          {isLoading && (
            <div className="mt-8 text-center">
              <div role="status" className="inline-block">
                  <svg aria-hidden="true" className="w-10 h-10 text-gray-200 animate-spin dark:text-gray-600 fill-primary-600" viewBox="0 0 100 101" fill="none" xmlns="http://www.w3.org/2000/svg">
                      <path d="M100 50.5908C100 78.2051 77.6142 100.591 50 100.591C22.3858 100.591 0 78.2051 0 50.5908C0 22.9766 22.3858 0.59082 50 0.59082C77.6142 0.59082 100 22.9766 100 50.5908ZM9.08144 50.5908C9.08144 73.1895 27.4013 91.5094 50 91.5094C72.5987 91.5094 90.9186 73.1895 90.9186 50.5908C90.9186 27.9921 72.5987 9.67226 50 9.67226C27.4013 9.67226 9.08144 27.9921 9.08144 50.5908Z" fill="currentColor"/>
                      <path d="M93.9676 39.0409C96.393 38.4038 97.8624 35.9116 97.0079 33.5539C95.2932 28.8227 92.871 24.3692 89.8167 20.348C85.8452 15.1192 80.8826 10.7238 75.2124 7.41289C69.5422 4.10194 63.2754 1.94025 56.7698 1.05124C51.7666 0.367541 46.6976 0.446843 41.7345 1.27873C39.2613 1.69328 37.813 4.19778 38.4501 6.62326C39.0873 9.04874 41.5694 10.4717 44.0505 10.1071C47.8511 9.54855 51.7191 9.52689 55.5402 10.0492C60.8642 10.7766 65.9928 12.5457 70.6331 15.2552C75.2735 17.9648 79.3347 21.5619 82.5849 25.841C84.9175 28.9121 86.7997 32.2913 88.1811 35.8758C89.083 38.2158 91.5421 39.6781 93.9676 39.0409Z" fill="currentFill"/>
                  </svg>
                  <span className="sr-only">Loading...</span>
              </div>
              <p className="mt-4 text-lg font-semibold text-primary-600 dark:text-primary-400">Applying Vector Quantization & analyzing...</p>
              <p className="text-sm text-gray-500 dark:text-gray-400">Our expert AI is carefully examining your image.</p>
            </div>
          )}

          {!isLoading && analysisResult && compressionStats && originalImage && processedImage && (
            <div className="mt-8">
              <ResultsDisplay
                originalImage={originalImage}
                processedImage={processedImage}
                stats={compressionStats}
                result={analysisResult}
                compressionApplied={compressionApplied}
                visualExplanation={visualExplanation}
                isLoadingExplanation={isLoadingExplanation}
              />
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
