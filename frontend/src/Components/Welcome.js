import React from 'react';
import { Link } from 'react-router-dom';
import Preloader from './Preloader';
import '../assets/css/welcomePage.css'; // Import CSS file for styling

const WelcomePage = () => {
    return (
        <div>
            <div className="background-animation">
                <div className="outer-circle">
                    <div className="wrapper">
                        <div className="breath">
                            <div className="flare1"></div>
                            <div className="flare2"></div>
                        </div>
                    </div>
                </div>
            </div>
            <Preloader />
            <div className="content-container">
                <div className="sign-container">
                    <h1 className="sign">CARE CUBE</h1>
            </div>
                <div className="center-content">
                    <div className='subtitle'>Connect-Embrace-Transform</div>
                    <br></br>
                    <br></br>
                    <Link to="/lor" className="get-started-link">
                        <button className="glow-on-hover">Get Started</button>
                    </Link>
                </div>
            </div>
        </div>
    );
}

export default WelcomePage;
