import './App.css';
import { Routes, Route, BrowserRouter, Link } from "react-router-dom";
import Login from './Components/Login';
import Register from './Components/Register';
import Home from './Components/Home';
import LorR from './Components/LorR';
import RegisCouncelling from './Components/RegisCouncelling';
import Welcome from './Components/Welcome';
import Game from './Components/Game';
import img1 from "./assets/img/back.png";
import img2 from "./assets/img/background.jpeg";
import CardProfile from './Components/CardProfile'
import Therapy from './Components/Therapy'
import Chat from './Components/Chat'
import New from './Components/New'
import Slider from './Components/slider';
import Game1 from './Components/Game1'

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route exact path="/login" element={<Login />} />
        <Route exact path="/register" element={<Register />} />
        <Route exact path="/" element={<Welcome />} />
        <Route exact path="/welcome" element={<Welcome />} />
        <Route exact path="/lor" element={<LorR />} />
        <Route exact path="/regisCouncelling" element={<RegisCouncelling />} />
        <Route exact path="/game" element={<Game />} />
        <Route exact path="/music" element={<CardProfile />} />
        <Route exact path="/Therapy" element={<Therapy />} />
        <Route exact path="/home/:id" element={<Home />} />
        <Route exact path="/chat" element={<Chat />} />
        <Route exact path="/summa" element={<New />} />
        <Route exact path="/home/slider" element={<Slider />} />
        <Route exact path="/home/game" element={<Game1 />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
