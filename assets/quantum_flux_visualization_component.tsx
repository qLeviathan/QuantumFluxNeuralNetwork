import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Slider } from '@/components/ui/slider'; 
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Settings, Play, Pause, RotateCcw, Eye } from 'lucide-react';

const QuantumStateVisualizer = () => {
  // Sample quantum states showing evolution through layers
  const [states, setStates] = useState(() => {
    const numLayers = 3;
    const numTokens = 10;
    
    return Array.from({ length: numLayers }, (_, layerIdx) => {
      // Create spiral pattern with increasingly chaotic positions in deeper layers
      return Array.from({ length: numTokens }, (_, tokenIdx) => {
        const baseAngle = (tokenIdx / numTokens) * 2 * Math.PI;
        const baseRadius = 0.3 + 0.6 * (tokenIdx / (numTokens - 1));
        
        // Add some randomness in later layers
        const noiseScale = layerIdx * 0.15;
        const angleNoise = (Math.random() - 0.5) * noiseScale * Math.PI;
        const radiusNoise = (Math.random() - 0.5) * noiseScale * 0.3;
        
        const angle = baseAngle + angleNoise;
        const radius = Math.max(0.1, Math.min(1.0, baseRadius + radiusNoise));
        
        return {
          rx: radius * Math.cos(angle),
          ry: radius * Math.sin(angle),
          r: radius,
          theta: angle,
          tokenId: tokenIdx
        };
      });
    });
  });
  
  // Simulation state
  const [currentLayer, setCurrentLayer] = useState(0);
  const [isAnimating, setIsAnimating] = useState(false);
  const [showLabels, setShowLabels] = useState(true);
  const [showVectors, setShowVectors] = useState(true);
  const [highlightToken, setHighlightToken] = useState(null);
  
  // Animation effect
  useEffect(() => {
    let animationTimer;
    if (isAnimating) {
      animationTimer = setInterval(() => {
        setCurrentLayer(prevLayer => (prevLayer + 1) % states.length);
      }, 1500);
    }
    return () => clearInterval(animationTimer);
  }, [isAnimating, states.length]);
  
  // Attention matrix calculation (simplified for visualization)
  const calculateAttention = (tokens) => {
    const matrix = [];
    for (let i = 0; i < tokens.length; i++) {
      const row = [];
      for (let j = 0; j < tokens.length; j++) {
        // Skip future tokens (causal masking)
        if (j > i) {
          row.push(0);
          continue;
        }
        
        // Calculate similarity: r_i * r_j * cos(θ_i - θ_j)
        const dot = tokens[i].rx * tokens[j].rx + tokens[i].ry * tokens[j].ry;
        const similarity = (dot + 1) / 2; // Scale to [0, 1]
        row.push(similarity);
      }
      matrix.push(row);
    }
    return matrix;
  };
  
  const attention = calculateAttention(states[currentLayer]);
  
  // SVG dimensions and scaling
  const svgSize = 400;
  const centerX = svgSize / 2;
  const centerY = svgSize / 2;
  const scale = svgSize * 0.4;
  
  return (
    <Card className="w-full max-w-4xl">
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center justify-between">
          <span>Quantum State Visualization</span>
          <div className="flex space-x-2">
            <Button 
              variant="outline" 
              size="icon" 
              onClick={() => setIsAnimating(!isAnimating)}
            >
              {isAnimating ? <Pause size={16} /> : <Play size={16} />}
            </Button>
            <Button 
              variant="outline" 
              size="icon" 
              onClick={() => setCurrentLayer(0)}
            >
              <RotateCcw size={16} />
            </Button>
            <Button 
              variant="outline" 
              size="icon" 
              onClick={() => setShowLabels(!showLabels)}
            >
              <Eye size={16} />
            </Button>
          </div>
        </CardTitle>
        <div className="flex items-center space-x-4">
          <span className="text-sm text-gray-500 w-16">Layer: {currentLayer}</span>
          <Slider 
            value={[currentLayer]} 
            min={0} 
            max={states.length - 1} 
            step={1} 
            onValueChange={(value) => setCurrentLayer(value[0])}
            className="flex-1"
          />
        </div>
      </CardHeader>
      
      <Tabs defaultValue="quantum" className="w-full">
        <TabsList className="w-full">
          <TabsTrigger value="quantum" className="flex-1">Quantum States</TabsTrigger>
          <TabsTrigger value="attention" className="flex-1">Attention Pattern</TabsTrigger>
        </TabsList>
        
        <TabsContent value="quantum" className="flex justify-center p-4">
          <svg width={svgSize} height={svgSize} viewBox={`0 0 ${svgSize} ${svgSize}`}>
            {/* Grid background */}
            <defs>
              <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
                <path d="M 40 0 L 0 0 0 40" fill="none" stroke="gray" strokeWidth="0.5" strokeOpacity="0.2" />
              </pattern>
            </defs>
            <rect width="100%" height="100%" fill="url(#grid)" />
            
            {/* Coordinate axes */}
            <line x1={0} y1={centerY} x2={svgSize} y2={centerY} stroke="gray" strokeWidth="1" strokeOpacity="0.5" />
            <line x1={centerX} y1={0} x2={centerX} y2={svgSize} stroke="gray" strokeWidth="1" strokeOpacity="0.5" />
            
            {/* Unit circle */}
            <circle cx={centerX} cy={centerY} r={scale} fill="none" stroke="blue" strokeWidth="1" strokeOpacity="0.3" />
            
            {/* Token vectors */}
            {showVectors && states[currentLayer].map((token, i) => (
              <line 
                key={`vector-${i}`}
                x1={centerX}
                y1={centerY}
                x2={centerX + token.rx * scale}
                y2={centerY - token.ry * scale}
                stroke={highlightToken === i ? "orange" : "blue"}
                strokeWidth={highlightToken === i ? 2 : 1}
                strokeOpacity="0.6"
              />
            ))}
            
            {/* Token points */}
            {states[currentLayer].map((token, i) => (
              <g key={`token-${i}`}>
                <circle 
                  cx={centerX + token.rx * scale} 
                  cy={centerY - token.ry * scale} 
                  r={highlightToken === i ? 8 : 6}
                  fill={highlightToken === i ? "orange" : `hsl(${token.tokenId * 36}, 70%, 60%)`}
                  strokeWidth={1}
                  stroke="white"
                  onMouseEnter={() => setHighlightToken(i)}
                  onMouseLeave={() => setHighlightToken(null)}
                />
                
                {showLabels && (
                  <text 
                    x={centerX + token.rx * scale + 10} 
                    y={centerY - token.ry * scale} 
                    fontSize="12" 
                    fill={highlightToken === i ? "orange" : "black"}
                  >
                    {token.tokenId}
                  </text>
                )}
              </g>
            ))}
            
            {/* Layer information */}
            <text x="10" y="20" fontSize="14" fill="black">
              Layer: {currentLayer} {currentLayer === 0 ? "(Input)" : ""}
            </text>
          </svg>
        </TabsContent>
        
        <TabsContent value="attention" className="p-4">
          <div className="overflow-auto">
            <div className="grid grid-cols-11 gap-1 w-full">
              {/* Column headers */}
              <div className="bg-gray-100 p-2 text-center font-bold">Token</div>
              {states[currentLayer].map((_, i) => (
                <div key={`col-${i}`} className="bg-gray-100 p-2 text-center font-bold">{i}</div>
              ))}
              
              {/* Attention matrix */}
              {attention.map((row, i) => (
                <React.Fragment key={`row-${i}`}>
                  <div className="bg-gray-100 p-2 text-center font-bold">{i}</div>
                  {row.map((value, j) => (
                    <div 
                      key={`cell-${i}-${j}`}
                      className="p-2 text-center text-xs"
                      style={{ 
                        backgroundColor: `rgba(0, 0, 255, ${value})`,
                        color: value > 0.5 ? 'white' : 'black'
                      }}
                      onMouseEnter={() => setHighlightToken(i)}
                      onMouseLeave={() => setHighlightToken(null)}
                    >
                      {value.toFixed(2)}
                    </div>
                  ))}
                </React.Fragment>
              ))}
            </div>
          </div>
        </TabsContent>
      </Tabs>
      
      <CardContent className="pt-2">
        <div className="text-sm text-gray-600">
          <p><strong>Quantum State Space:</strong> Tokens are represented as points in a 2D quantum space where:</p>
          <ul className="list-disc pl-5 mt-2 space-y-1">
            <li>Radius (r): Token importance - larger radius = more important</li>
            <li>Angle (θ): Token semantics - similar meanings have similar angles</li>
            <li>Attention strength: Based on geometric proximity of token states</li>
            <li>Evolution: States evolve through physical equations inspired by quantum mechanics</li>
          </ul>
        </div>
      </CardContent>
    </Card>
  );
};

export default QuantumStateVisualizer;
