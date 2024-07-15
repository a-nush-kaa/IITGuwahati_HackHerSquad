document.getElementById('categoryForm').addEventListener('submit', function(event) {
    event.preventDefault();
    
    // Get selected category
    const category = document.getElementById('category').value;
    
    // Fetch recommendations for the selected category
    fetch(`/recommendations?category=${category}`)
        .then(response => response.json())
        .then(data => {
            displayRecommendations(data);
        })
        .catch(error => console.error('Error fetching recommendations:', error));
});

function displayRecommendations(recommendations) {
    const recommendationsDiv = document.getElementById('recommendations');
    recommendationsDiv.innerHTML = ''; // Clear previous recommendations
    
    recommendations.forEach(item => {
        const itemCard = document.createElement('div');
        itemCard.classList.add('recommendation-item');
        
        // Add image
        const img = document.createElement('img');
        img.src = item.image_url;
        img.alt = item.description;
        itemCard.appendChild(img);
        
        // Add description
        const description = document.createElement('p');
        description.textContent = item.description;
        itemCard.appendChild(description);
        
        recommendationsDiv.appendChild(itemCard);
    });
}
