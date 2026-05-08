// --- Chart Objects (Global) ---
let dashboardChart, moistureReportChart, waterReportChart;

// --- Flag to prevent multiple fetches at once ---
let isFetching = false;

// --- Global reference to the update interval ---
let updateInterval;

// --- Main Function ---
document.addEventListener('DOMContentLoaded', function () {
    // --- Select Elements ---
    const navItems = document.querySelectorAll('.nav-item');
    const pages = document.querySelectorAll('.page');
    const pageTitle = document.getElementById('page-title');
    
    // All UI elements
    const ui = {
        soilMoisture: document.getElementById('dashboard-soil-moisture'),
        pumpStatus: document.getElementById('dashboard-pump-status'),
        
        pumpToggle: document.getElementById('pump-toggle'),
        pumpToggleStatus: document.getElementById('pump-toggle-status'),
        
        modeToggle: document.getElementById('mode-toggle'),
        modeToggleStatus: document.getElementById('mode-toggle-status'),
        
        arduinoStatusText: document.getElementById('arduino-status-text'),
        arduinoStatusIcon: document.querySelector('#arduino-status-indicator .status-icon'),
        
        systemArduino: document.getElementById('system-status-arduino'),
        systemPump: document.getElementById('system-status-pump'),
        
        themeToggle: document.getElementById('theme-toggle'),
        themeToggleStatus: document.getElementById('theme-toggle-status'),
        
        profileIcon: document.getElementById('profile-icon'),
        profileDropdown: document.getElementById('profile-dropdown'),
        logoutButton: document.getElementById('logout-button')
    };

    // --- Page Navigation (Robust Fix) ---
    navItems.forEach(item => {
        item.addEventListener('click', () => {
            navItems.forEach(nav => nav.classList.remove('active'));
            item.classList.add('active');
            const pageId = item.getAttribute('data-page');
            
            // 1. Try to find the span first (Best way)
            const navTextSpan = item.querySelector('.nav-text');
            
            if (navTextSpan) {
                pageTitle.textContent = navTextSpan.textContent;
            } else {
                // 2. Fallback: Get all text and remove the icon name using Regex
                let fullText = item.textContent.trim();
                let cleanText = fullText.replace(/^[a-z_]+\s*/, ''); 
                pageTitle.textContent = cleanText;
            }

            showPage(pageId);
        });
    });

    function showPage(pageId) {
        pages.forEach(page => page.classList.remove('active'));
        const targetPage = document.getElementById(pageId);
        if (targetPage) {
            targetPage.classList.add('active');
        }
    }

    // --- Theme Toggle ---
    function applyTheme(theme) {
        if (theme === 'light') {
            document.body.classList.add('light-mode');
            ui.themeToggle.checked = false;
            ui.themeToggleStatus.textContent = 'Current: Light Mode';
        } else {
            document.body.classList.remove('light-mode');
            ui.themeToggle.checked = true;
            ui.themeToggleStatus.textContent = 'Current: Dark Mode';
        }
    }

    ui.themeToggle.addEventListener('change', () => {
        const theme = ui.themeToggle.checked ? 'dark' : 'light';
        localStorage.setItem('theme', theme);
        applyTheme(theme);
    });

    // Load saved theme
    const savedTheme = localStorage.getItem('theme') || 'dark';
    applyTheme(savedTheme);

    // --- Profile Dropdown ---
    ui.profileIcon.addEventListener('click', (event) => {
        event.stopPropagation();
        ui.profileDropdown.classList.toggle('show');
    });

    ui.logoutButton.addEventListener('click', () => {
        window.location.href = '/logout';
    });

    window.addEventListener('click', (event) => {
        if (ui.profileDropdown.classList.contains('show') && !ui.profileDropdown.contains(event.target)) {
            ui.profileDropdown.classList.remove('show');
        }
    });

    // --- API Communication ---

    async function updateData() {
        if (isFetching) return;
        isFetching = true;

        try {
            const response = await fetch('/api/data');
            if (response.status === 401) {
                window.location.href = '/login'; 
                return;
            }
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            
            const state = await response.json();

            // Update Connection Status
            if (state.connected) {
                ui.arduinoStatusText.textContent = "Connected";
                ui.arduinoStatusIcon.classList.remove('error');
                ui.arduinoStatusIcon.classList.add('ok');
                ui.arduinoStatusIcon.innerHTML = "✓";
                ui.systemArduino.innerHTML = '<span class="status-dot ok"></span> Arduino';
            } else {
                ui.arduinoStatusText.textContent = "Not Connected";
                ui.arduinoStatusIcon.classList.remove('ok');
                ui.arduinoStatusIcon.classList.add('error');
                ui.arduinoStatusIcon.innerHTML = "!";
                ui.systemArduino.innerHTML = '<span class="status-dot error"></span> Arduino';
            }

            // Update Dashboard Cards
            ui.soilMoisture.textContent = state.data.soil_moisture + '%';
            ui.pumpStatus.textContent = state.data.pump_status;
            ui.pumpStatus.className = `card-metric-value ${state.data.pump_status === 'ON' ? 'status-on' : 'status-off'}`;
            
            // --- Update Toggles ---
            const pumpIsOn = state.data.pump_status === "ON";
            const modeIsAuto = state.data.mode === "AUTO";

            ui.pumpToggle.checked = pumpIsOn;
            ui.modeToggle.checked = modeIsAuto;

            if (modeIsAuto) {
                ui.modeToggleStatus.textContent = "Mode: AUTO";
                ui.pumpToggle.disabled = true; 
                ui.pumpToggleStatus.textContent = `Status: ${state.data.pump_status} (Auto)`;
                ui.modeToggle.disabled = false;
            } else {
                ui.modeToggleStatus.textContent = "Mode: MANUAL";
                ui.pumpToggle.disabled = false;
                ui.modeToggle.disabled = false;
                ui.pumpToggleStatus.textContent = `Status: ${state.data.pump_status}`;
            }
            
            // Update System Status Pump
            const pumpDotClass = pumpIsOn ? 'ok' : 'error';
            ui.systemPump.innerHTML = `<span class="status-dot ${pumpDotClass}"></span> Pump`;
            
            // Update Dashboard Chart
            updateDashboardChart(state.data.soil_moisture);
            
        } catch (error) {
            console.error('Error fetching data:', error);
        } finally {
            isFetching = false;
        }
    }

    async function sendCommand(url) {
        clearInterval(updateInterval);
        isFetching = true; 
        
        try {
            await fetch(url, { method: 'POST' });
            await new Promise(resolve => setTimeout(resolve, 500)); 
            await updateData();
        } catch (error) {
            console.error('Error sending command:', error);
        }
        
        isFetching = false; 
        updateInterval = setInterval(updateData, 2000);
    }
    
    // --- Toggle Event Listeners ---
    
    ui.modeToggle.addEventListener('change', async () => {
        ui.modeToggle.disabled = true;
        ui.pumpToggle.disabled = true;
        
        if (ui.modeToggle.checked) {
            await sendCommand('/api/pump/auto');
        } else {
            await sendCommand('/api/pump/off');
        }
    });

    ui.pumpToggle.addEventListener('change', async () => {
        ui.modeToggle.disabled = true;
        ui.pumpToggle.disabled = true;

        if (ui.pumpToggle.checked) {
            await sendCommand('/api/pump/on');
        } else {
            await sendCommand('/api/pump/off');
        }
    });


    // --- Charting ---
    function createDashboardChart() {
        const ctx = document.getElementById('dashboardMoistureChart').getContext('2d');
        dashboardChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Soil Moisture %',
                    data: [],
                    fill: false,
                    borderColor: 'rgb(56, 189, 248)', // Fixed Color
                    tension: 0.4,
                    pointBackgroundColor: 'rgb(56, 189, 248)', // Fixed Color
                    pointRadius: 0,
                }]
            },
            options: chartOptions
        });
    }

    function createReportCharts() {
        const moistureCtx = document.getElementById('moistureChart').getContext('2d');
        moistureReportChart = new Chart(moistureCtx, {
            type: 'line',
            data: {
                labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                datasets: [{
                    label: 'Soil Moisture %',
                    data: [58, 60, 62, 40, 65, 62, 75], 
                    fill: false,
                    borderColor: 'rgb(56, 189, 248)', // Fixed Color
                    tension: 0.4,
                }]
            },
            options: chartOptions
        });
        
        const waterUsageCtx = document.getElementById('waterUsageChart').getContext('2d');
        waterReportChart = new Chart(waterUsageCtx, {
            type: 'bar',
            data: {
                labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                datasets: [{
                    label: 'Liters',
                    data: [25, 20, 30, 22, 35, 18, 28], 
                    backgroundColor: 'rgba(56, 189, 248, 0.5)', // Fixed Color
                    borderColor: 'rgba(56, 189, 248, 1)', // Fixed Color
                    borderWidth: 1
                }]
            },
            options: chartOptions
        });
    }

    // Common options for all charts
    const chartOptions = {
        scales: {
            y: {
                beginAtZero: true,
                grid: { color: 'rgba(255, 255, 255, 0.1)' },
                ticks: { color: 'var(--text-secondary)' }
            },
            x: {
                grid: { color: 'rgba(255, 255, 255, 0.1)' },
                ticks: { color: 'var(--text-secondary)' }
            }
        },
        plugins: {
            legend: {
                display: false
            }
        },
        maintainAspectRatio: false
    };
    
    function updateDashboardChart(value) {
        if (!dashboardChart) return;
        const chart = dashboardChart;
        const now = new Date();
        chart.data.labels.push(now.toLocaleTimeString());
        chart.data.datasets[0].data.push(value);
        if(chart.data.labels.length > 20) {
            chart.data.labels.shift();
            chart.data.datasets[0].data.shift();
        }
        chart.update();
    }
    
    // --- Start Application ---
    createDashboardChart();
    createReportCharts(); 
    updateData(); 
    updateInterval = setInterval(updateData, 2000); 
    
    document.querySelector('.nav-item[data-page="dashboard"]').click();
});


