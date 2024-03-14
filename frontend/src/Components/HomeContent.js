import React, { useState } from 'react';
import PuzzleImage from '../assets/img/puzzle.jpg';
import ChatBot from '../assets/img/bot.jpg';
import Yoga from '../assets/img/yoga.jpg';
import Depression from '../assets/img/depression.jpg';
import Game from './Game';
import Game1 from './Game1';
import Preloader from './Preloader';
import { Link } from 'react-router-dom';

const HomeContent = ({ activeButton }) => {
    const [gameClicked, setGameClicked] = useState(false);

    return (
        <>
            <Preloader />
            {!gameClicked && 
                <>  
                    <div className="project-box-wrapper">
                        <div className="project-box" style={{ backgroundColor: '#ffff' }}>
                            <img src={PuzzleImage} style={{ width:'100%', height:"90%" }} alt="Puzzle" />
                            <div className="project-box-footer">
                                <div className="days-left" style={{ color: '#df3670' }}>
                                    <span onClick={() => setGameClicked(true)} style={{ cursor: "pointer" }}>Game</span >
                                </div>
                            </div>
                        </div> 
                    </div>
                    <div className="project-box-wrapper" style={{marginTop:"120px"}}>
                        <div className="project-box" style={{ backgroundColor: '#fff', height:"100%" }} >
                            <img src={ChatBot} style={{ width:'100%', height:"80%" }} alt="ChatBot" />
                            <Link to={'/chat'}>
                                <div className="project-box-footer">
                                    <div className="days-left" style={{ color: '#df3670' }}>
                                        <span style={{ cursor:"pointer" }}>ChatBot</span>
                                    </div>
                                </div>
                            </Link>
                        </div>
                    </div>
                    <div className="project-box-wrapper">
                        <div className="project-box" style={{ backgroundColor: '#ffff', height:"100%" }} >
                            <img src={Yoga} style={{ width:'100%', height:"80%" }} alt="Yoga" />
                            <div className="project-box-footer">
                                <div className="days-left" style={{ color: '#df3670' }}>
                                    <span style={{ cursor:"pointer" }}>Yoga</span>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div className="project-box-wrapper">
                        <div className="project-box" style={{ backgroundColor: '#ffff', height:"100%" }} >
                            <img src={Depression} style={{ width:'100%', height:"80%" }} alt="Depression" />
                            <div className="project-box-footer">
                                <div className="days-left" style={{ color: '#df3670' }}>
                                    <span style={{ cursor:"pointer" }}>Depression</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </>
            }
            {gameClicked && 
                <>
                    <h2 style={{ color:"white", fontWeight:"bold" }}>Game</h2>
                    <Game1 />
                    <button onClick={() => setGameClicked(false)} style={{ backgroundColor:"#000", color:"#007bff", fontSize:"30px", fontWeight:"bold" }}>Back</button>
                </>
            }
        </>
    );
}

export default HomeContent;
