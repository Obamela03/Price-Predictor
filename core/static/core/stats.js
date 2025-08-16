// stats.js

const avgPriceCtx = document.getElementById('avgPriceChart').getContext('2d');
const approvalCtx = document.getElementById('approvalChart').getContext('2d');

const avgLabels = JSON.parse(document.getElementById('avgLabels').textContent);
const avgData = JSON.parse(document.getElementById('avgData').textContent);
const approvalData = JSON.parse(document.getElementById('approvalData').textContent);

new Chart(avgPriceCtx, {
    type: 'bar',
    data: {
        labels: avgLabels,
        datasets: [{
            label: 'Avg Price (R)',
            data: avgData,
            backgroundColor: '#007bff'
        }]
    }
});

new Chart(approvalCtx, {
    type: 'pie',
    data: {
        labels: ['Approved', 'Pending'],
        datasets: [{
            data: approvalData,
            backgroundColor: ['#28a745', '#ffc107']
        }]
    }
});
