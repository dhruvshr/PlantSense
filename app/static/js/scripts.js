function submitChat(event) {
    event.preventDefault();
    
    const formData = new FormData(event.target);
    const chatLog = document.querySelector('.chat-log');
    
    // Add user message immediately
    const userMessage = document.createElement('p');
    userMessage.className = 'user-message';
    userMessage.innerHTML = formData.get('user_feedback');
    chatLog.appendChild(userMessage);
    
    // Clear input field
    event.target.reset();
    
    // Scroll to the new message
    userMessage.scrollIntoView({ behavior: 'smooth' });
    
    fetch('/chat', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // Add bot response
            const botMessage = document.createElement('p');
            botMessage.className = 'bot-message';
            botMessage.innerHTML = data.conversation[data.conversation.length - 1].message;
            chatLog.appendChild(botMessage);
            
            // Scroll to bot response
            botMessage.scrollIntoView({ behavior: 'smooth' });
        } else {
            console.error('Error:', data.error);
        }
    })
    .catch(error => console.error('Error:', error));
}

// Click to scroll functionality
document.addEventListener('DOMContentLoaded', function() {
    const chatLog = document.querySelector('.chat-log');
    if (chatLog) {
        chatLog.addEventListener('click', function(event) {
            const message = event.target.closest('.user-message, .bot-message');
            if (message) {
                message.scrollIntoView({ behavior: 'smooth' });
            }
        });
    }
});

