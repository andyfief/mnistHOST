import React, {useState} from 'react';
import Canvas from './components/Canvas';
import PredictionResult from './components/PredictionResult';
import ScrollDownPtr from './components/ScrollDownPtr';
import CodeSnippet from './components/Snippet';
import{ predictDigit } from './services/api';
import './styles/App.css';

function App() {
  const [prediction, setPrediction] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (imageData) => {
    setIsLoading(true);
    setError(null);

    try{
      const result = await predictDigit(imageData);
       if (result.predicted_digit === 'none') {
        setPrediction(null);
      } else {
        setPrediction(result);
  }
    } catch (error) {
      setError('Error predicting digit. Please try again');
      console.error(error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="page-container">
      <div className="App">
        <div className="canvas-screen">
          <Canvas onSubmit={handleSubmit} />
        </div>
        <div className="prediction-screen">
          {!prediction && <div className="prediction_welcome">Draw a digit and <br />click "Predict" <br /> ... </div>}
          {prediction && <PredictionResult result={prediction} />}
        </div>
        <div className="background-image">
          <div className="scrollDown">
              <ScrollDownPtr />
          </div>
        </div>
        {error && <p className="error">{error}</p>}
    </div>
      <div className="information-section">
          <h3 id='tips'>Tips</h3>
          <p className = 'tipsLines'> Draw a large number.</p>
          <p className = 'tipsLines'> Use a straight |, without a base.</p>
          <p className = 'tipsLines'> Use an empty 0, without the slash.</p>
          <h3>About the MNIST Dataset</h3>
          <p>The MNIST database is a large collection of handwritten digits used for training various image processing systems.</p>
          <p>It contains 70,000 images of handwritten digits (0-9), where each image is a 28×28 grayscale pixel grid. </p>
          <p>This page is an interface for a neural network written from scratch in Python - no frameworks like PyTorch or TensorFlow were used.</p>
          <p> The submitted image is processed to match the dataset more accurately, and is then fed through the neural network to make a prediction.</p>
          <p> The neural network's accuracy after being tested on the dataset is 92%.</p>
      </div>
      <div className="snippet">
        <h3>Model Training Source Code</h3>
        <span className="githublink">
          Or visit the{' '}
          <a href="https://github.com/andyfief/MNIST-from-scratch" style={{ textDecoration: 'underline' }}>
            GitHub Repository
          </a>
        </span>
            <CodeSnippet />
      </div>
    </div>
  );
}

export default App;
