import React, { useEffect, useState } from 'react'
import axios from 'axios'
import './Home.css'
import { Link } from 'react-router-dom'
import { useLocation, useNavigate } from 'react-router-dom';
import HomeContent from './HomeContent';
import RegisCouncelling from './RegisCouncelling'
import Analysis from './Analysis'
import AboutUs from './AboutUs';
import Profile from '../assets/img/profile-user.png'
import Logo from '../assets/img/carecube.png'
import Preloader from './Preloader';
import Therapy from './Therapy';

const Home = () => {
  const [user,setUser] =useState(null)
  const userId=(useLocation().pathname).split('/')[2];
  const [activeButton, setActiveButton] = useState("homecontent");
  const navigate = useNavigate()
  useEffect(()=>{
    const fetchUser = async () => {
      try{
        console.log("hello");
      const user = await axios.post('http://localhost:4000/getUser',{
        userId:userId
      });
      console.log("user: ",user);
      console.log(user.data.detail);
      setUser(user.data.detail)
      }catch(err){
        console.log(err);
      }
    }
    fetchUser()
  },[userId])
  useEffect(()=>{
    console.log(user);
  },[user])
      const handleButtonClick = (buttonName) => {
        setActiveButton(buttonName);
        if (buttonName === "logout") {
          navigate("/login");
        }
      };
      
  return (
         <div className="app-container" style={{marginTop:"10px" ,width:"100%",height:"100%"}}>
          {/* <Preloader /> */}
          
          <div className="app-header" style={{width:"100%",height:"15vh"}} >
            <div className='app-header-left'>
              <img src={Logo} style={{width:"140px",height:"100px",marginTop:"15px"}}></img>
            </div>
            <div className='app-header-right'>
            <img src={Profile} alt="" style={{width:"50px"}}/>
            <div>{user && user.username}</div>
            </div>
          </div>
          
          <div className="app-content" style={{width:"100%",height:"85vh"}}>
            <div className="app-sidebar">
            <button
                className={activeButton === "homecontent" ? "active" : ""}
                onClick={() => handleButtonClick("homecontent")}
                >
                Home
                </button>
              <button
                className={activeButton === "therapy" ? "active" : ""}
                onClick={() => handleButtonClick("therapy")}
                >
                Therapy
                </button>
           
              <button
                className={activeButton === "councelling" ? "active" : ""}
                onClick={() => handleButtonClick("councelling")}
                >
                Counselling
                </button>
                <button
                className={activeButton === "analysis" ? "active" : ""}
                onClick={() => handleButtonClick("analysis")}
                >
                Analysis
                </button>
                <button
                className={activeButton === "aboutus" ? "active" : ""}
                onClick={() => handleButtonClick("aboutus")}
                >
                About Us
                </button>
              
              
            </div>
            <div className="projects-section" style={{}}>
              <div className="projects-section-header">
                <p></p>
                <p className="time"></p>
              </div>
              
              <div className="project-boxes jsGridView"  style={{ width:"850px",height:"700px",marginBottom:"60px"}}>
               
                {activeButton==="homecontent" &&  <HomeContent activeButton={activeButton}/>}
                {activeButton==="councelling" &&  <RegisCouncelling/>}
                {activeButton==="analysis" &&  <Analysis/>}
                {activeButton==="aboutus" &&  <AboutUs/>}
                {activeButton==="therapy" &&  <Therapy/>}
                
              </div>
            </div>
                
              </div>
            </div>
            
        
  
  )
}

export default Home