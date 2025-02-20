import { Box, Container, Typography, ThemeProvider, createTheme } from '@mui/material';
import ImageClassifier from './components/ImageClassifier';

const theme = createTheme({
  palette: {
    primary: {
      main: '#1976d2',
    },
  },
});

function App() {
  return (
    <ThemeProvider theme={theme}>
      <Container>
        <Box sx={{ width: '100vw', my: 4, textAlign: 'center', display: 'flex', flexDirection: 'column', alignItems: 'center', 'justifyContent': 'center' }}>
          <Typography variant="h3" component="h1" gutterBottom>
            Guitar Handedness Detector
          </Typography>
          <Typography variant="h6" color="text.secondary" paragraph>
            Upload or paste an image of a guitar to detect if it's left-handed or right-handed
          </Typography>
          <ImageClassifier modelPath="/handedness_model.onnx" />
        </Box>
      </Container>
    </ThemeProvider>
  );
}

export default App;
