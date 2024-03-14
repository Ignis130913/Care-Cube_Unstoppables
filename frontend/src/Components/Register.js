import React, { useState } from 'react';
import axios from 'axios';
import { useNavigate, Link } from 'react-router-dom';
import Preloader from './Preloader';

const Register = () => {
    const navigate = useNavigate();
    const [formData, setFormData] = useState({
        name: '',
        email: '',
        password: '',
        age: '',
        gender: 'male',
        familyBackground: 'rich',
        goals: '',
        currentStatus: '',
        phone: ''
    });

    const handleChange = (e) => {
        const { name, value } = e.target;
        setFormData({ ...formData, [name]: value });
    };

    const handleSubmit = async (event) => {
        event.preventDefault();
        try {
            const response = await axios.post(
                "http://localhost:4000/register",
                formData,
                { withCredentials: true }
            );
            const res = response.data;
            if (res.status === true) {
                navigate('/welcome');
            } else {
                alert(res.message);
                setFormData({
                    name: '',
                    email: '',
                    password: '',
                    age: '',
                    gender: 'male',
                    familyBackground: 'rich',
                    goals: '',
                    currentStatus: '',
                    phone: ''
                });
            }
        } catch (ex) {
            console.log(ex);
        }
    };

    return (
        <div style={{
            position: 'relative',
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center',
            minHeight: '100vh',
        }}>
            <Preloader />
            <div style={{
                width: '90%',
                maxWidth: '800px',
                padding: '20px',
                backgroundColor: 'rgba(255, 255, 255, 0.8)',
                borderRadius: '10px',
                boxShadow: '0 0 10px rgba(0, 0, 0, 0.1)',
                textAlign: 'center',
            }}>
                <h2>Register</h2>
                <form onSubmit={handleSubmit}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '20px' }}>
                        <div style={{ flexBasis: '45%' }}>
                            <label htmlFor="name">Name:</label>
                            <input type="text" id="name" name="name" value={formData.name} onChange={handleChange} />
                        </div>
                        <div style={{ flexBasis: '45%' }}>
                            <label htmlFor="email">Email:</label>
                            <input type="email" id="email" name="email" value={formData.email} onChange={handleChange} />
                        </div>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '20px' }}>
                        <div style={{ flexBasis: '45%' }}>
                            <label htmlFor="password">Password:</label>
                            <input type="password" id="password" name="password" value={formData.password} onChange={handleChange} />
                        </div>
                        <div style={{ flexBasis: '45%' }}>
                            <label htmlFor="age">Age:</label>
                            <input type="number" id="age" name="age" value={formData.age} onChange={handleChange} />
                        </div>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '20px' }}>
                        <div style={{ flexBasis: '45%' }}>
                            <label htmlFor="gender">Gender:</label>
                            <select id="gender" name="gender" value={formData.gender} onChange={handleChange}>
                                <option value="male">Male</option>
                                <option value="female">Female</option>
                                <option value="other">Other</option>
                            </select>
                        </div>
                        <div style={{ flexBasis: '45%' }}>
                            <label htmlFor="family-background">Family Background:</label>
                            <select id="family-background" name="familyBackground" value={formData.familyBackground} onChange={handleChange}>
                                <option value="rich">Rich</option>
                                <option value="poor">Poor</option>
                            </select>
                        </div>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '20px' }}>
                        <div style={{ flexBasis: '45%' }}>
                            <label htmlFor="goals">Goals:</label>
                            <textarea id="goals" name="goals" value={formData.goals} onChange={handleChange}></textarea>
                        </div>
                        <div style={{ flexBasis: '45%' }}>
                            <label htmlFor="current-status">Current Status:</label>
                            <textarea id="current-status" name="currentStatus" value={formData.currentStatus} onChange={handleChange}></textarea>
                        </div>
                    </div>
                    <div>
                        <label htmlFor="phone">Phone Number:</label>
                        <input type="tel" id="phone" name="phone" value={formData.phone} onChange={handleChange} />
                    </div>
                    <Link to={'/login'}><button class="button-64" role="button"><span class="text">Register</span></button></Link>
                </form>
            </div>
        </div>
    );
};

export default Register;
