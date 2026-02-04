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
        if (blob) onChange(blob);
      }, "image/png");
    }
  };

  return (
    <div className="canvas-container">
      <SignatureCanvas
        penColor="white"
        backgroundColor="black"
        minWidth={12}
        maxWidth={16}
        canvasProps={{ width: 280, height: 280, className: "canvas" }}
        ref={sigCanvas}
        onEnd={handleEnd}
      />
    </div>
  );
};

export default CanvasDrawComponent;
