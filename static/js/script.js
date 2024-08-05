setInterval(() => {
    fetch('/status')
        .then(response => response.json())
        .then(data => {
            document.getElementById('status').textContent = data.recognized ? "Face Matched" : "Face Not Matched";
            document.getElementById('status').style.color = data.recognized ? "green" : "red";
        })
        .catch(error => console.error('Error fetching status:', error));
}, 1000);