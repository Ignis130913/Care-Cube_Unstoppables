import React, { useState } from 'react';
import axios from 'axios';
import './Chat.css'

function Chat() {
    const [chatHistory, setChatHistory] = useState([]);
    const [userInput, setUserInput] = useState('');

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!userInput.trim()) return;

        // Display user message
        setChatHistory(prevChatHistory => [...prevChatHistory, { user: true, message: userInput }]);

        try {
            // Send message to backend using Axios
            const response = await axios.post('http://localhost:5000/get_response', {
                user_input: userInput
            });

            // Display bot response in a separate message box
            setChatHistory(prevChatHistory => [...prevChatHistory, { user: false, message: response.data.bot_response }]);
        } catch (error) {
            console.error('Error fetching response:', error);
            // Optionally handle error here, e.g., display error message to user
        }

        // Clear input field
        setUserInput('');
    };

    return (
        <div className="ChatApp">
            <h1 className="ChatHeader">CHAT WITH CARE BOT</h1>
            <div className="ChatContainer">
                {chatHistory.map((item, index) => (
                    <div key={index} className={item.user ? 'UserMessage' : 'BotMessage'}>
                        {item.message}
                    </div>
                ))}
            </div>
            <form onSubmit={handleSubmit} className="InputForm">
                <input
                    type="text"
                    value={userInput}
                    onChange={(e) => setUserInput(e.target.value)}
                    placeholder="Type your message..."
                    className="MessageInput"
                />
                <button type="submit" className="SendButton">Send</button>
            </form>
        </div>
    );
}

export default Chat;

