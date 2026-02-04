// frontend/src/App.js
import React, { useRef, useState, useCallback } from "react";
import Webcam from "react-webcam";
import axios from "axios";
import "./App.css"; // We will keep the default styling for now

function App() {
  // 1. Create a "reference" to the webcam so we can take screenshots
  const webcamRef = useRef(null);
  
  // 2. Create "State" variables to store the app's status
  // result: stores the answer from Python (e.g., "Face detected")
  const [result, setResult] = useState(null); 
  // imageSrc: stores the snapshot we take
  const [imageSrc, setImageSrc] = useState(null);

  // 3. Define the "Capture" function
  // useCallback is a React hook that prevents the function from being recreated every render
  const capture = useCallback(async () => {
    
    // A. Take a screenshot from the webcam
    const imageData = webcamRef.current.getScreenshot();
    setImageSrc(imageData); // Save it to show on screen

    try {
      // B. Send the image to Python using Axios
      // We are sending a POST request to your Flask server
      const response = await axios.post("https://face-recognition-app-k4ww.onrender.com/predict", {
        image: imageData,
      });

      // C. Save the response from Python to our state
      console.log("Response from Python:", response.data);
      setResult(response.data);

    } catch (error) {
      console.error("Error connecting to Python:", error);
      setResult({ message: "Error connecting to server" });
    }
  }, [webcamRef]);

  return (
    <div style={styles.container}>
      <h1>Face Recognition System</h1>
      
      <div style={styles.webcamBox}>
        {/* The Webcam Component */}
        <Webcam
          audio={false}
          ref={webcamRef}
          screenshotFormat="image/jpeg"
          width={500}
        />
      </div>

      {/* The Button to trigger the Capture function */}
      <button onClick={capture} style={styles.button}>
        Scan My Face
      </button>

      {/* Display the Results if we have them */}
      {imageSrc && (
        <div style={styles.resultBox}>
          <h3>Last Snapshot:</h3>
          <img src={imageSrc} alt="Captured" width="200" />
        </div>
      )}

      {result && (
        <div style={styles.resultBox}>
          <h3>Result from AI:</h3>
          <p>Message: <b>{result.message}</b></p>
          <p>Faces Found: <b>{result.face_count}</b></p>
        </div>
      )}
    </div>
  );
}

// Simple CSS styles directly in the file for simplicity
const styles = {
  container: {
    textAlign: "center",
    padding: "20px",
    fontFamily: "Arial, sans-serif",
  },
  webcamBox: {
    margin: "20px auto",
    border: "5px solid #333",
    width: "fit-content",
  },
  button: {
    padding: "10px 20px",
    fontSize: "18px",
    backgroundColor: "#007BFF",
    color: "white",
    border: "none",
    cursor: "pointer",
    borderRadius: "5px",
  },
  resultBox: {
    marginTop: "20px",
    padding: "10px",
    border: "1px solid #ccc",
    display: "inline-block",
    textAlign: "left",
  }
};

export default App;