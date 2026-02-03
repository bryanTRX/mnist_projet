import React, { useRef, useEffect } from "react";
import SignatureCanvas from "react-signature-canvas";

const CanvasDrawComponent = ({ onChange, resetSignal }) => {
  const sigCanvas = useRef(null);

  useEffect(() => {
    if (resetSignal && sigCanvas.current) {
      sigCanvas.current.clear();
    }
  }, [resetSignal]);

  const handleEnd = () => {
    if (sigCanvas.current) {
      const canvas = sigCanvas.current.getCanvas(); 
      canvas.toBlob((blob) => {
        onChange(blob);
      }, "image/png");
    }
  };

  return (
    <div style={{ border: "1px solid black", width: 280, height: 280 }}>
      <SignatureCanvas
        penColor="white"
        backgroundColor="black"
        canvasProps={{ width: 280, height: 280 }}
        ref={sigCanvas}
        onEnd={handleEnd}
      />
    </div>
  );
};

export default CanvasDrawComponent;
