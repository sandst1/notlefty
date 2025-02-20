import { useState, useRef, useEffect } from 'react';
import { Box, Typography, Paper, CircularProgress, Button } from '@mui/material';
import * as ort from 'onnxruntime-web';

interface ImageClassifierProps {
  modelPath: string;
}

export default function ImageClassifier({ modelPath }: ImageClassifierProps) {
  const [image, setImage] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [session, setSession] = useState<ort.InferenceSession | null>(null);
  const dropZoneRef = useRef<HTMLDivElement>(null);

  // Initialize ONNX Runtime session
  useEffect(() => {
    async function initONNX() {
      try {
        const session = await ort.InferenceSession.create(modelPath);
        setSession(session);
      } catch (error) {
        console.error('Failed to load ONNX model:', error);
      }
    }
    initONNX();
  }, [modelPath]);

  // Image preprocessing function
  const preprocessImage = async (imageData: ImageData): Promise<Float32Array> => {
    const tensor = new Float32Array(1 * 3 * 224 * 224);
    const { data } = imageData;
    
    // Normalize and resize (assuming the model expects 224x224 images)
    for (let i = 0; i < 224 * 224; i++) {
      const red = data[i * 4] / 255;
      const green = data[i * 4 + 1] / 255;
      const blue = data[i * 4 + 2] / 255;
      
      // Normalize using ImageNet stats
      tensor[i] = (red - 0.485) / 0.229;
      tensor[i + 224 * 224] = (green - 0.456) / 0.224;
      tensor[i + 2 * 224 * 224] = (blue - 0.406) / 0.225;
    }
    
    return tensor;
  };

  // Prediction function
  const predict = async (imageElement: HTMLImageElement) => {
    if (!session) return;

    const canvas = document.createElement('canvas');
    canvas.width = 224;
    canvas.height = 224;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Draw and resize image
    ctx.drawImage(imageElement, 0, 0, 224, 224);
    const imageData = ctx.getImageData(0, 0, 224, 224);
    const tensor = await preprocessImage(imageData);

    // Run inference
    const inputTensor = new ort.Tensor('float32', tensor, [1, 3, 224, 224]);
    const outputMap = await session.run({ input: inputTensor });
    const output = outputMap.output.data as Float32Array;

    // Get prediction
    const prediction = output[0] > output[1] ? 'Left Handed' : 'Right Handed';
    return `${prediction}`;
  };

  // Handle file drop
  const handleDrop = async (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      setLoading(true);
      setPrediction(null);
      
      const reader = new FileReader();
      reader.onload = async (e) => {
        const dataUrl = e.target?.result as string;
        setImage(dataUrl);
        
        const img = new Image();
        img.onload = async () => {
          const result = await predict(img);
          setPrediction(result || 'Failed to predict');
          setLoading(false);
        };
        img.src = dataUrl;
      };
      reader.readAsDataURL(file);
    }
  };

  // Handle paste
  const handlePaste = async (e: ClipboardEvent) => {
    const items = e.clipboardData?.items;
    if (!items) return;

    for (const item of items) {
      if (item.type.startsWith('image/')) {
        const file = item.getAsFile();
        if (file) {
          setLoading(true);
          setPrediction(null);
          
          const reader = new FileReader();
          reader.onload = async (e) => {
            const dataUrl = e.target?.result as string;
            setImage(dataUrl);
            
            const img = new Image();
            img.onload = async () => {
              const result = await predict(img);
              setPrediction(result || 'Failed to predict');
              setLoading(false);
            };
            img.src = dataUrl;
          };
          reader.readAsDataURL(file);
        }
      }
    }
  };

  // Add reset function
  const handleReset = () => {
    setImage(null);
    setPrediction(null);
  };

  useEffect(() => {
    document.addEventListener('paste', handlePaste);
    return () => document.removeEventListener('paste', handlePaste);
  }, [session]);

  return (
    <Box sx={{ maxWidth: 600, mx: 'auto', p: 2 }}>
      <Paper
        ref={dropZoneRef}
        onDrop={handleDrop}
        onDragOver={(e) => e.preventDefault()}
        sx={{
          p: 4,
          textAlign: 'center',
          cursor: 'pointer',
          border: '2px dashed #ccc',
          '&:hover': { borderColor: 'primary.main' },
        }}
      >
        <Typography variant="h6" gutterBottom>
          Drop an image here or paste from clipboard
        </Typography>
        {loading && <CircularProgress sx={{ my: 2 }} />}
        {image && (
          <>
            <Box sx={{ mt: 2 }}>
              <img
                src={image}
                alt="Uploaded guitar"
                style={{ maxWidth: '100%', maxHeight: 300, objectFit: 'contain' }}
              />
            </Box>
            <Button 
              variant="contained" 
              onClick={handleReset}
              sx={{ mt: 2 }}
            >
              Reset
            </Button>
          </>
        )}
        {prediction && (
          <Typography variant="h5" sx={{ mt: 2, fontWeight: 'bold' }}>
            {prediction}
          </Typography>
        )}
      </Paper>
    </Box>
  );
} 