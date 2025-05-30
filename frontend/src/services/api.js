const API_URL = '/api'; 

export const predictDigit = async (imageData) => {
    try {
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({image: imageData}),
        });

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return await response.json();
    } catch(error) {
        console.error('Error predicting digit:', error);
        throw error;
    }
};