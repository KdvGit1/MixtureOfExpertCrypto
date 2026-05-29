/**
 * ⚡ Cyberpunk Canvas Particle Engine
 * Highly optimized, CPU-friendly GPU-accelerated graphics
 */

const canvas = document.getElementById("particlesCanvas");
const ctx = canvas.getContext("2d");

let particles = [];
let width = (canvas.width = window.innerWidth);
let height = (canvas.height = window.innerHeight);

// Handle window resizing
window.addEventListener("resize", () => {
    width = canvas.width = window.innerWidth;
    height = canvas.height = window.innerHeight;
});

class Particle {
    constructor(x, y, color, size, speedX, speedY, maxLife) {
        this.x = x;
        this.y = y;
        this.color = color;
        this.size = size;
        this.speedX = speedX;
        this.speedY = speedY;
        this.maxLife = maxLife;
        this.life = maxLife;
    }

    update() {
        this.x += this.speedX;
        this.y += this.speedY;
        this.life--;
    }

    draw() {
        const opacity = this.life / this.maxLife;
        ctx.save();
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
        ctx.fillStyle = this.color;
        ctx.shadowBlur = 10;
        ctx.shadowColor = this.color;
        ctx.globalAlpha = opacity;
        ctx.fill();
        ctx.restore();
    }
}

// Background ambient loop
function spawnAmbientParticles() {
    if (particles.length < 30 && Math.random() < 0.1) {
        // Spawn small floating cyan pixels
        const x = Math.random() * width;
        const y = height + 10;
        const size = Math.random() * 2 + 1;
        const speedX = (Math.random() - 0.5) * 0.4;
        const speedY = -Math.random() * 0.6 - 0.3;
        const life = Math.random() * 200 + 100;
        particles.push(new Particle(x, y, "#00f0ff", size, speedX, speedY, life));
    }
}

// Sparkle Burst Alerts
function triggerSignalBurst(type, x, y) {
    let color = "#00ff66"; // BUY default
    let count = 40;
    
    if (type === "SELL") {
        color = "#ff007f";
    } else if (type === "HOLD") {
        color = "#00f0ff";
        count = 15;
    } else if (type === "STREAK") {
        color = "#ffb700";
        count = 50;
    }

    for (let i = 0; i < count; i++) {
        const angle = Math.random() * Math.PI * 2;
        const speed = Math.random() * 6 + 2;
        const size = Math.random() * 3 + 1.5;
        const life = Math.random() * 40 + 30;
        
        const speedX = Math.cos(angle) * speed;
        // Rising sparks for BUY, falling embers for SELL
        let speedY = Math.sin(angle) * speed;
        if (type === "BUY") speedY -= 1.5;
        if (type === "SELL") speedY += 1.5;

        particles.push(
            new Particle(
                x,
                y,
                color,
                size,
                speedX,
                speedY,
                life
            )
        );
    }
}

// Main graphics rendering loop
function render() {
    ctx.clearRect(0, 0, width, height);

    spawnAmbientParticles();

    for (let i = particles.length - 1; i >= 0; i--) {
        const p = particles[i];
        p.update();
        p.draw();

        if (p.life <= 0) {
            particles.splice(i, 1);
        }
    }

    requestAnimationFrame(render);
}

// Export functions to window
window.triggerSignalBurst = triggerSignalBurst;

// Start graphics loop
render();
