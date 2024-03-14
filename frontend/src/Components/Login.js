import React,{useState,useEffect} from 'react'
import { Link, useNavigate } from "react-router-dom";
import axios from 'axios'
import Preloader from './Preloader';

const Login = () => {
    const navigate = useNavigate();
    const [values, setValues] = useState({ username: "", password: "" });
    const handleSubmit = async (event) => {
        event.preventDefault();
        try {
          const response = await axios.post(
            "http://localhost:4000/login",
            {
              ...values,
            },
            { withCredentials: true }
          );    
          const res = response.data;
            if(res.status===true){
                navigate(`/home/${res.user._id}`)
            }else{
                alert(res.message);
                setValues({username:"",password:""})
            }
        } catch (ex) {
          console.log(ex);
        }
    }
  return (
    <div className="container">
      <Preloader />
      <div className="content">
        <h2>Login</h2>
        <form onSubmit={(e) => handleSubmit(e)}>
          <div className='input-group'>
            <label htmlFor="username">Username</label>
            <input
              type="username"
              name="username"
              onChange={(e) =>
                setValues({ ...values, [e.target.name]: e.target.value })
              }
            />
          </div>
          <div className='input-group'>
            <label htmlFor="password">Password</label>
            <input
              type="password"
              name="password"
              onChange={(e) =>
                setValues({ ...values, [e.target.name]: e.target.value })
              }
            />
          </div>
          <button class="button-64" role="button" type='submit'><span class="text">Login</span></button>
        </form>
      </div>
    </div>

   
  )
}

export default Login