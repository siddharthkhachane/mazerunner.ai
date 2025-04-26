import React, { useState, useEffect, useRef } from 'react';
import * as tf from '@tensorflow/tfjs';

export default function MazeSolverBob() {
  const canvasWidth = 400;
  const canvasHeight = 400;
  const gridSize = 30; 
  const columns = Math.floor(canvasWidth / gridSize);
  const rows = Math.floor(canvasHeight / gridSize);
  
  const DRAWING = 'drawing';
  const TRAINING = 'training';
  const PLAYING = 'playing';
  
  const WALL = 'wall';
  const START = 'start';
  const GOAL = 'goal';
  const ERASER = 'eraser';
  
  const [gameState, setGameState] = useState(DRAWING);
  const [tool, setTool] = useState(WALL);
  const [maze, setMaze] = useState(Array(rows).fill().map(() => Array(columns).fill(0))); 
  const [start, setStart] = useState(null);
  const [goal, setGoal] = useState(null);
  const [agent, setAgent] = useState(null);
  const [episode, setEpisode] = useState(0);
  const [score, setScore] = useState(0);
  const [isDrawing, setIsDrawing] = useState(false);
  const [memory, setMemory] = useState([]);
  const [path, setPath] = useState([]);
  const [trainComplete, setTrainComplete] = useState(false);
  
  const canvasRef = useRef(null);
  const animationRef = useRef(null);
  const agentRef = useRef(null);
  const modelRef = useRef(null);
  const targetNetworkRef = useRef(null);
  
  const gamma = 0.95; 
  const epsilon = 0.3; 
  const learningRate = 0.005; 
  const batchSize = 16; 
  const targetUpdateFreq = 5; 
  const memorySize = 1000;
  const hiddenSize = 64; 
  
  const actions = [
    [-1, 0], 
    [0, 1],  
    [1, 0], 
    [0, -1] 
  ];
  
  useEffect(() => {
    drawMaze();
  }, [maze, start, goal, agent, path]);
  
  const createModel = () => {
    const model = tf.sequential();
    model.add(tf.layers.dense({
      units: hiddenSize,
      activation: 'relu',
      inputShape: [6], 
      kernelInitializer: 'heNormal' 
    }));
    model.add(tf.layers.dense({
      units: hiddenSize,
      activation: 'relu',
      kernelInitializer: 'heNormal'
    }));
    model.add(tf.layers.dense({
      units: 4, 
      activation: 'linear',
      kernelInitializer: 'glorotNormal'
    }));
    
    model.compile({
      optimizer: tf.train.adam(learningRate),
      loss: 'meanSquaredError'
    });
    
    return model;
  };
  
  const cloneModel = (model) => {
    const clone = tf.sequential();
    const weights = model.getWeights();
    const layerConfigs = [];
    
    for (const layer of model.layers) {
      layerConfigs.push(layer.getConfig());
    }
    
    for (let i = 0; i < layerConfigs.length; i++) {
      const layer = tf.layers[layerConfigs[i].className].fromConfig(layerConfigs[i]);
      clone.add(layer);
    }
    
    clone.setWeights(weights);
    
    return clone;
  };
  
  const drawMaze = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvasWidth, canvasHeight);
    
    ctx.strokeStyle = '#ddd';
    ctx.lineWidth = 0.5;
    
    for (let i = 0; i <= rows; i++) {
      ctx.beginPath();
      ctx.moveTo(0, i * gridSize);
      ctx.lineTo(canvasWidth, i * gridSize);
      ctx.stroke();
    }
    
    for (let i = 0; i <= columns; i++) {
      ctx.beginPath();
      ctx.moveTo(i * gridSize, 0);
      ctx.lineTo(i * gridSize, canvasHeight);
      ctx.stroke();
    }
    
    ctx.fillStyle = '#333';
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < columns; c++) {
        if (maze[r][c] === 1) {
          ctx.fillRect(c * gridSize, r * gridSize, gridSize, gridSize);
        }
      }
    }
    
    if (start) {
      ctx.fillStyle = '#4CAF50';
      ctx.beginPath();
      ctx.arc(
        start.col * gridSize + gridSize / 2,
        start.row * gridSize + gridSize / 2,
        gridSize / 3,
        0,
        Math.PI * 2
      );
      ctx.fill();
    }
    
    if (goal) {
      ctx.fillStyle = '#E91E63';
      ctx.beginPath();
      ctx.arc(
        goal.col * gridSize + gridSize / 2,
        goal.row * gridSize + gridSize / 2,
        gridSize / 3,
        0,
        Math.PI * 2
      );
      ctx.fill();
    }
    
    if (agent) {
      ctx.fillStyle = '#2196F3';
      ctx.beginPath();
      ctx.arc(
        agent.col * gridSize + gridSize / 2,
        agent.row * gridSize + gridSize / 2,
        gridSize / 2.5,
        0,
        Math.PI * 2
      );
      ctx.fill();
      
      ctx.fillStyle = 'white';
    }
    
    if (path.length > 0 && gameState === PLAYING) {
      ctx.strokeStyle = 'rgba(33, 150, 243, 0.4)';
      ctx.lineWidth = gridSize / 4;
      ctx.beginPath();
      ctx.moveTo(
        start.col * gridSize + gridSize / 2,
        start.row * gridSize + gridSize / 2
      );
      
      for (const pos of path) {
        ctx.lineTo(
          pos.col * gridSize + gridSize / 2,
          pos.row * gridSize + gridSize / 2
        );
      }
      
      ctx.stroke();
    }
  };
  
  const handleCanvasClick = (e) => {
    if (gameState !== DRAWING) return;
    
    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    const col = Math.floor(x / gridSize);
    const row = Math.floor(y / gridSize);
    
    if (row < 0 || row >= rows || col < 0 || col >= columns) return;
    
    const newMaze = [...maze];
    
    if (tool === WALL) {
      newMaze[row][col] = 1;
      setMaze(newMaze);
    } else if (tool === ERASER) {
      newMaze[row][col] = 0;
      setMaze(newMaze);
      
      if (start && start.row === row && start.col === col) {
        setStart(null);
      }
      
      if (goal && goal.row === row && goal.col === col) {
        setGoal(null);
      }
    } else if (tool === START) {
      if (newMaze[row][col] === 1) return;
      if (goal && goal.row === row && goal.col === col) return;
      
      setStart({ row, col });
    } else if (tool === GOAL) {
      if (newMaze[row][col] === 1) return;
      if (start && start.row === row && start.col === col) return;
      
      setGoal({ row, col });
    }
  };
  
  const handleMouseDown = (e) => {
    setIsDrawing(true);
    handleCanvasClick(e);
  };
  
  const handleMouseMove = (e) => {
    if (!isDrawing || gameState !== DRAWING) return;
    handleCanvasClick(e);
  };
  
  const handleMouseUp = () => {
    setIsDrawing(false);
  };
  
  const getState = (position) => {
    const { row, col } = position;
    const goalRow = goal.row;
    const goalCol = goal.col;
    
    const walls = [
      row > 0 ? maze[row - 1][col] : 1,           
      col < columns - 1 ? maze[row][col + 1] : 1, 
      row < rows - 1 ? maze[row + 1][col] : 1,    
      col > 0 ? maze[row][col - 1] : 1            
    ];
    
    const rowDiff = (goalRow - row) / rows;
    const colDiff = (goalCol - col) / columns;
    
    return [row / rows, col / columns, rowDiff, colDiff, ...walls];
  };
  
  const chooseAction = (state) => {
    if (Math.random() < epsilon) {
      return Math.floor(Math.random() * 4);
    } else {
      return tf.tidy(() => {
        const stateTensor = tf.tensor2d([state], [1, state.length]);
        const qValues = modelRef.current.predict(stateTensor);
        return qValues.argMax(1).dataSync()[0];
      });
    }
  };
  
  const takeAction = (position, actionIdx) => {
    const { row, col } = position;
    const [dRow, dCol] = actions[actionIdx];
    const newRow = row + dRow;
    const newCol = col + dCol;
    
    if (
      newRow < 0 ||
      newRow >= rows ||
      newCol < 0 ||
      newCol >= columns ||
      maze[newRow][newCol] === 1
    ) {
      return {
        position: { row, col }, 
        reward: -10,
        done: false
      };
    }
    
    const reachedGoal = newRow === goal.row && newCol === goal.col;
    
    return {
      position: { row: newRow, col: newCol },
      reward: reachedGoal ? 100 : -1, 
      done: reachedGoal
    };
  };
  
  const updateMemory = (state, action, reward, nextState, done) => {
    const newMemory = [...memory, { state, action, reward, nextState, done }];
    if (newMemory.length > memorySize) {
      newMemory.shift();
    }
    setMemory(newMemory);
  };
  
  const trainModel = async () => {
    if (memory.length < batchSize) return;
    
    const batch = [];
    const indices = [];
    
    while (indices.length < batchSize) {
      const idx = Math.floor(Math.random() * memory.length);
      if (!indices.includes(idx)) {
        indices.push(idx);
        batch.push(memory[idx]);
      }
    }
    
    const states = batch.map(exp => exp.state);
    const nextStates = batch.map(exp => exp.nextState);
    
    const qValues = tf.tidy(() => {
      const stateTensor = tf.tensor2d(states, [batchSize, states[0].length]);
      return modelRef.current.predict(stateTensor);
    });
    
    const nextQValues = tf.tidy(() => {
      const nextStateTensor = tf.tensor2d(nextStates, [batchSize, nextStates[0].length]);
      return targetNetworkRef.current.predict(nextStateTensor);
    });
    
    const qValuesData = await qValues.array();
    const nextQValuesData = await nextQValues.arraySync();
    
    for (let i = 0; i < batchSize; i++) {
      const { action, reward, done } = batch[i];
      
      if (done) {
        qValuesData[i][action] = reward;
      } else {
        const nextMaxQ = Math.max(...nextQValuesData[i]);
        qValuesData[i][action] = reward + gamma * nextMaxQ;
      }
    }
    
    const updatedQValues = tf.tensor2d(qValuesData, [batchSize, 4]);
    const stateTensor = tf.tensor2d(states, [batchSize, states[0].length]);
    
    await modelRef.current.fit(stateTensor, updatedQValues, {
      epochs: 1,
      verbose: 0
    });
    
    qValues.dispose();
    nextQValues.dispose();
    updatedQValues.dispose();
    stateTensor.dispose();
  };
  
  const updateTargetNetwork = () => {
    const weights = modelRef.current.getWeights();
    targetNetworkRef.current.setWeights(weights);
  };
  
  const resetGame = () => {
    if (start) {
      setAgent({ ...start });
      setPath([]);
      setScore(0);
    }
  };
  
  const startTraining = async () => {
    if (!start || !goal) {
      alert('Please place start and goal points');
      return;
    }
    
    if (!modelRef.current) {
      modelRef.current = createModel();
      targetNetworkRef.current = cloneModel(modelRef.current);
    }
    
    setGameState(TRAINING);
    setAgent({ ...start });
    setEpisode(0);
    setMemory([]);
    setPath([]);
    setTrainComplete(false);
    
 
    const bootstrapMemory = () => {
      for (let actionIdx = 0; actionIdx < 4; actionIdx++) {
        const [dRow, dCol] = actions[actionIdx];
        const newRow = start.row + dRow;
        const newCol = start.col + dCol;
        
        if (newRow < 0 || newRow >= rows || newCol < 0 || newCol >= columns) continue;
        
        if (maze[newRow][newCol] === 1) continue;
        
        const oldDist = Math.abs(start.row - goal.row) + Math.abs(start.col - goal.col);
        const newDist = Math.abs(newRow - goal.row) + Math.abs(newCol - goal.col);
        
        const reward = newDist < oldDist ? 5 : -5;
        const state = getState(start);
        const nextState = getState({ row: newRow, col: newCol });
        const done = newRow === goal.row && newCol === goal.col;
        
        updateMemory(state, actionIdx, reward, nextState, done);
      }
    };
    
    bootstrapMemory();
    
    const maxEpisodes = 100;
    const maxSteps = 100; 
    
    let stuckCounter = 0;
    let lastPosition = null;
    
    for (let ep = 0; ep < maxEpisodes; ep++) {
      let currentAgent = { ...start };
      let totalReward = 0;
      let currentPath = [];
      stuckCounter = 0;
      lastPosition = null;
      
      const currentEpsilon = Math.max(0.1, epsilon - (ep / maxEpisodes) * 0.2);
      
      for (let step = 0; step < maxSteps; step++) {
        const state = getState(currentAgent);
        
        const action = Math.random() < currentEpsilon ? 
                       Math.floor(Math.random() * 4) : // More exploration early on
                       tf.tidy(() => {
                         const stateTensor = tf.tensor2d([state], [1, state.length]);
                         const qValues = modelRef.current.predict(stateTensor);
                         return qValues.argMax(1).dataSync()[0];
                       });
        
        const { position: newPosition, reward, done } = takeAction(currentAgent, action);
        
        if (lastPosition && 
            lastPosition.row === newPosition.row && 
            lastPosition.col === newPosition.col) {
          stuckCounter++;
          
          if (stuckCounter > 5) {
            let randomAction;
            do {
              randomAction = Math.floor(Math.random() * 4);
            } while (randomAction === action);
            
            const result = takeAction(currentAgent, randomAction);
            newPosition.row = result.position.row;
            newPosition.col = result.position.col;
            stuckCounter = 0;
          }
        } else {
          stuckCounter = 0;
        }
        
        lastPosition = { ...newPosition };
        
        totalReward += reward;
        
        if (newPosition.row !== currentAgent.row || newPosition.col !== currentAgent.col) {
          currentPath.push(newPosition);
        }
        
        const nextState = getState(newPosition);
        
        updateMemory(state, action, reward, nextState, done);
        
        currentAgent = newPosition;
        
        setAgent(currentAgent);
        setPath(currentPath);
        
        if (memory.length >= batchSize) {
          const trainIterations = ep < 10 ? 3 : 1;
          for (let i = 0; i < trainIterations; i++) {
            await trainModel();
          }
        }
        
        if (step % targetUpdateFreq === 0) {
          updateTargetNetwork();
        }
        
        setScore(totalReward);
        
        if (done || step === maxSteps - 1) {
          break;
        }
        
        await new Promise(resolve => setTimeout(resolve, 10));
      }
      
      setEpisode(ep + 1);
      
      await new Promise(resolve => setTimeout(resolve, 50)); // Reduced delay
    }
    
    setTrainComplete(true);
    setGameState(PLAYING);
    resetGame();
  };
  
  const playGame = async () => {
    resetGame();
    
    let currentAgent = { ...start };
    let currentPath = [];
    let totalReward = 0;
    let done = false;
    let step = 0;
    const maxSteps = 100;
    let stuckCount = 0;
    let lastPosition = null;
    
    while (!done && step < maxSteps) {
      const state = getState(currentAgent);
      
      let action;
      
      if (modelRef.current && trainComplete) {
        action = tf.tidy(() => {
          const stateTensor = tf.tensor2d([state], [1, state.length]);
          const qValues = modelRef.current.predict(stateTensor);
          return qValues.argMax(1).dataSync()[0];
        });
      } else {
        const rowDiff = goal.row - currentAgent.row;
        const colDiff = goal.col - currentAgent.col;
        
        if (Math.abs(rowDiff) > Math.abs(colDiff)) {
          action = rowDiff > 0 ? 2 : 0; 
        } else {
          action = colDiff > 0 ? 1 : 3; 
        }
        
        if (stuckCount > 3) {
          action = (action + 1) % 4; 
          stuckCount = 0;
        }
      }
      
      const { position: newPosition, reward, done: newDone } = takeAction(currentAgent, action);
      
      if (lastPosition && 
          lastPosition.row === newPosition.row && 
          lastPosition.col === newPosition.col) {
        stuckCount++;
        
        if (stuckCount > 3) {
          let randomAction;
          do {
            randomAction = Math.floor(Math.random() * 4);
          } while (randomAction === action);
          
          const result = takeAction(currentAgent, randomAction);
          newPosition.row = result.position.row;
          newPosition.col = result.position.col;
        }
      } else {
        stuckCount = 0;
      }
      
      lastPosition = { ...newPosition };
      
      totalReward += reward;
      
      if (newPosition.row !== currentAgent.row || newPosition.col !== currentAgent.col) {
        currentPath.push(newPosition);
      }
      
      currentAgent = newPosition;
      
      setAgent(currentAgent);
      setPath(currentPath);
      
      setScore(totalReward);
      
      done = newDone;
      step++;
      
      await new Promise(resolve => setTimeout(resolve, 100));
    }
  };
  
  const resetAll = () => {
    setGameState(DRAWING);
    setMaze(Array(rows).fill().map(() => Array(columns).fill(0)));
    setStart(null);
    setGoal(null);
    setAgent(null);
    setEpisode(0);
    setScore(0);
    setPath([]);
    setTrainComplete(false);
    if (modelRef.current) {
      modelRef.current.dispose();
      modelRef.current = null;
    }
    if (targetNetworkRef.current) {
      targetNetworkRef.current.dispose();
      targetNetworkRef.current = null;
    }
  };
  
  return (
    <div className="flex flex-col items-center p-4 space-y-4 w-full bg-gray-100">
      <h1 className="text-2xl font-bold text-gray-800"> Maze Runner</h1>
      
      <div className="flex flex-col md:flex-row md:space-x-4 space-y-4 md:space-y-0 w-full justify-center">
      <div className="bg-white p-4 rounded-lg shadow-md">
          <canvas
            ref={canvasRef}
            width={canvasWidth}
            height={canvasHeight}
            className="border border-gray-300 bg-white"
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseUp}
          />
          
          {gameState === DRAWING && (
            <div className="flex flex-col mt-4 space-y-4">
              <div className="flex space-x-2">
                <button
                  className={`px-3 py-2 rounded-md ${tool === WALL ? 'bg-blue-500 text-white' : 'bg-gray-200'}`}
                  onClick={() => setTool(WALL)}
                >
                  Wall
                </button>
                <button
                  className={`px-3 py-2 rounded-md ${tool === START ? 'bg-green-500 text-white' : 'bg-gray-200'}`}
                  onClick={() => setTool(START)}
                >
                  Start
                </button>
                <button
                  className={`px-3 py-2 rounded-md ${tool === GOAL ? 'bg-pink-500 text-white' : 'bg-gray-200'}`}
                  onClick={() => setTool(GOAL)}
                >
                  Goal
                </button>
                <button
                  className={`px-3 py-2 rounded-md ${tool === ERASER ? 'bg-red-500 text-white' : 'bg-gray-200'}`}
                  onClick={() => setTool(ERASER)}
                >
                  Erase
                </button>
              </div>
              
              <div className="flex space-x-2">
                <button
                  className="px-3 py-2 bg-purple-600 text-white rounded-md"
                  onClick={playGame}
                  disabled={!start || !goal}
                >
                  Run
                </button>
                <button
                  className="px-3 py-2 bg-gray-600 text-white rounded-md"
                  onClick={resetAll}
                >
                  Reset All
                </button>
              </div>
            </div>
          )}
          
          {gameState !== DRAWING && (
            <div className="flex flex-col mt-4 space-y-4">
              <div className="flex space-x-2">
                <button
                  className="px-3 py-2 bg-green-600 text-white rounded-md"
                  onClick={playGame}
                >
                  Show Solution
                </button>
                <button
                  className="px-3 py-2 bg-gray-600 text-white rounded-md"
                  onClick={resetAll}
                >
                  Reset All
                </button>
              </div>
            </div>
          )}
        </div>
        
        <div className="flex flex-col space-y-4">
          <div className="bg-white p-4 rounded-lg shadow-md">
            <h2 className="text-xl font-semibold mb-2">Stats</h2>
            <div className="space-y-2">
              <p>Mode: <span className="font-medium">{gameState}</span></p>
              <p>Episodes: <span className="font-medium">{episode}</span></p>
              <p>Score: <span className="font-medium">{score}</span></p>
              {trainComplete && <p className="text-green-600 font-semibold">Training Complete!</p>}
            </div>
          </div>
          
          <div className="bg-white p-4 rounded-lg shadow-md">
            <h2 className="text-xl font-semibold mb-2">Instructions</h2>
            <ol className="list-decimal list-inside space-y-2 text-sm">
              <li>Draw walls using the Wall tool</li>
              <li>Place a start point and a goal point</li>
              <li>Click "Start Training" to let the blob learn</li>
              <li>After training, click "Show Solution" to see the path</li>
            </ol>
          </div>
          
          <div className="bg-white p-4 rounded-lg shadow-md">
            <h2 className="text-xl font-semibold mb-2">About</h2>
            <ul className="list-disc list-inside space-y-2 text-sm">
              <li>Tiny DQN model (64-64 dense layers)</li>
              <li>Rewards: -1 per step, -10 hitting walls, +100 reaching goal</li>
              <li>Fully client-side implementation</li>
              <li>No backend required</li>
              <li>TensorFlow.js powered</li>
            </ul>
          </div>
          
          <div className="bg-white p-4 rounded-lg shadow-md">
            <h2 className="text-xl font-semibold mb-2">Status</h2>
            <div className="text-sm">
              {agent && start && goal && (
                <div>
                  <p>Distance to goal: {Math.abs(agent.row - goal.row) + Math.abs(agent.col - goal.col)} cells</p>
                  <p>Current action: {path.length > 0 ? 
                    path[path.length-1].row > path[path.length-2]?.row ? "Down" : 
                    path[path.length-1].row < path[path.length-2]?.row ? "Up" : 
                    path[path.length-1].col > path[path.length-2]?.col ? "Right" : "Left" 
                    : "None"}</p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}