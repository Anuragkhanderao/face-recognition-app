import React, { useState, useRef, useCallback } from "react";
import Webcam from "react-webcam";
import axios from "axios";
import "./App.css";

function App() {
  const webcamRef = useRef(null);
  const [imgSrc, setImgSrc] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);

  const capture = useCallback(async () => {
    const imageSrc = webcamRef.current.getScreenshot();
    setImgSrc(imageSrc);
    setLoading(true); // Start loading animation
    setPrediction(null); // Clear old result

    try {
      // Send to Backend
      const response = await axios.post("/predict", {
        image: imageSrc,
      });

      setPrediction(response.data);
    } catch (error) {
      console.error("Error:", error);
      setPrediction({ message: "Error connecting to server" });
    } finally {
      setLoading(false); // Stop loading animation
    }
  }, [webcamRef]);

  return (
    <div className="app-container">
      {/* Navbar */}
      <header className="navbar">
        <div className="logo">👁️ SecureScan AI</div>
        <div className="status-dot"></div>
      </header>

      <main className="main-content">
        <div className="scanner-section">
          <div className="camera-frame">
            <Webcam
              audio={false}
              ref={webcamRef}
              screenshotFormat="image/jpeg"
              className="webcam-feed"
            />
            {/* Overlay for scanning effect */}
            <div className="scan-overlay"></div>
          </div>

          <button 
            className="capture-btn" 
            onClick={capture} 
            disabled={loading}
          >
            {loading ? "Scanning..." : "Identify User"}
          </button>
        </div>

        {/* Results Panel */}
        <div className="results-section">
          {imgSrc && (
            <div className="result-card fade-in">
              <h3>Captured Snapshot</h3>
              <img src={imgSrc} alt="captured" className="captured-img" />
              
              <div className="prediction-box">
                {loading ? (
                  <div className="spinner"></div>
                ) : (
                  <>
                    <h4>AI Analysis:</h4>
                    <p className="result-text">
                      {prediction ? prediction.message : "Processing..."}
                    </p>
                  </>
                )}
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

export default App;