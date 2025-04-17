document.addEventListener('DOMContentLoaded', function () {
    const predictBtn = document.getElementById('submit');
    const outputDiv = document.getElementById('output');

    if (predictBtn && outputDiv) {
        predictBtn.addEventListener('click', function () {
            setTimeout(() => {
                outputDiv.scrollIntoView({ behavior: 'smooth' });
            }, 300); 
        });
    }
})