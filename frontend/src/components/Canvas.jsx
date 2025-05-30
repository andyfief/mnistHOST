import React, { useState, useRef } from 'react';

function Canvas({ onSubmit }) {
  const canvasRef = useRef(null);
  const [isDrawing, setIsDrawing] = useState(false);

  const width = 28;
  const height = 28;

  React.useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = 'rgb(93, 212, 248)';
    ctx.fillRect(0, 0, width, height);
  }, []);

  const getRelativeCoords = (clientX, clientY) => {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const scaleX = width / rect.width;
    const scaleY = height / rect.height;

    return {
      x: (clientX - rect.left) * scaleX,
      y: (clientY - rect.top) * scaleY
    };
  };

  const startDrawing = (x, y) => {
    const ctx = canvasRef.current.getContext('2d');
    ctx.beginPath();
    ctx.moveTo(x, y);
    setIsDrawing(true);
  };

  const draw = (x, y) => {
    if (!isDrawing) return;

    const ctx = canvasRef.current.getContext('2d');
    ctx.lineWidth = 2;
    ctx.lineCap = 'round';
    ctx.strokeStyle = 'black';
    ctx.lineTo(x, y);
    ctx.stroke();
  };

  // Mouse Handlers
  const handleMouseDown = (e) => {
    const { x, y } = getRelativeCoords(e.clientX, e.clientY);
    startDrawing(x, y);
  };

  const handleMouseMove = (e) => {
    const { x, y } = getRelativeCoords(e.clientX, e.clientY);
    draw(x, y);
  };

  // Touch Handlers
  const handleTouchStart = (e) => {
    const touch = e.touches[0];
    const { x, y } = getRelativeCoords(touch.clientX, touch.clientY);
    startDrawing(x, y);
  };

  const handleTouchMove = (e) => {
    const touch = e.touches[0];
    const { x, y } = getRelativeCoords(touch.clientX, touch.clientY);
    draw(x, y);
  };

  const stopDrawing = () => setIsDrawing(false);

  const clearCanvas = () => {
    const ctx = canvasRef.current.getContext('2d');
    ctx.fillStyle = 'rgb(93, 212, 248)';
    ctx.fillRect(0, 0, width, height);
  };

  const getImageData = () => {
    const dataURL = canvasRef.current.toDataURL('image/png');
    const base64Image = dataURL.split(',')[1];
    onSubmit(base64Image);
  };

  return (
    <div className='canvas-container'>
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        style={{
          border: '2px dashed rgb(19, 0, 233)',
          width: `${width * 5}px`,
          height: `${height * 5}px`,
          imageRendering: 'pixelated',
          touchAction: 'none' // Prevents scroll while drawing
        }}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={stopDrawing}
        onMouseLeave={stopDrawing}
        onTouchStart={handleTouchStart}
        onTouchMove={handleTouchMove}
        onTouchEnd={stopDrawing}
      />
      <div style={{ marginTop: '3px' }}>
        <button onClick={clearCanvas} style={{ color: 'rgb(19, 0, 233)', backgroundColor: 'rgb(255, 255, 255)', border: '2px solid rgb(19, 0, 233)', marginRight: '2px', fontFamily: 'Courier New, monospace' }}>Clear</button>
        <button onClick={getImageData} style={{ color: 'rgb(19, 0, 233)', backgroundColor: 'rgb(255, 255, 255)', border: '2px solid rgb(19, 0, 233)', marginLeft: '2px', fontFamily: 'Courier New, monospace' }}>Predict</button>
      </div>
    </div>
  );
}

export default Canvas;
