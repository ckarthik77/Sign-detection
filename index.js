
// Smooth scrolling for navigation links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// Navbar scroll effect
window.addEventListener('scroll', () => {
    const navbar = document.getElementById('navbar');
    const progressBar = document.getElementById('progressBar');
    
    if (window.scrollY > 50) {
        navbar.classList.add('scrolled');
    } else {
        navbar.classList.remove('scrolled');
    }

    // Update progress bar
    const winScroll = document.body.scrollTop || document.documentElement.scrollTop;
    const height = document.documentElement.scrollHeight - document.documentElement.clientHeight;
    const scrolled = (winScroll / height) * 100;
    progressBar.style.width = scrolled + '%';
});

// Intersection Observer for fade-in animations
const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.classList.add('visible');
        }
    });
}, observerOptions);

// Observe all fade-in elements
document.querySelectorAll('.fade-in').forEach(el => {
    observer.observe(el);
});

// Animated counters for stats
function animateCounter(element, target) {
    let current = 0;
    const increment = target / 100;
    const timer = setInterval(() => {
        current += increment;
        if (current >= target) {
            current = target;
            clearInterval(timer);
        }
        element.textContent = Math.floor(current).toLocaleString();
    }, 20);
}

// Start counter animations when stats section is visible
const statsObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            const counters = entry.target.querySelectorAll('.stat-number');
            counters.forEach(counter => {
                const target = parseInt(counter.dataset.target);
                animateCounter(counter, target);
            });
            statsObserver.unobserve(entry.target);
        }
    });
}, { threshold: 0.5 });

const statsSection = document.querySelector('.stats');
if (statsSection) {
    statsObserver.observe(statsSection);
}

// Demo button with loading state
document.getElementById('demoBtn').addEventListener('click', function(e) {
    e.preventDefault();
    const btn = this;
    const originalText = btn.innerHTML;
    
    btn.innerHTML = '<span class="loading"></span> Loading Demo...';
    btn.style.pointerEvents = 'none';
    
    setTimeout(() => {
        btn.innerHTML = originalText;
        btn.style.pointerEvents = 'auto';
        alert('🚀 Demo would open here! This is a preview version.');
    }, 2000);
});

// Add floating particles animation
function createFloatingParticle() {
    const particle = document.createElement('div');
    const randomX = Math.random();
    
    particle.style.cssText = `
        position: fixed;
        width: 4px;
        height: 4px;
        background: rgba(0, 212, 255, 0.6);
        border-radius: 50%;
        pointer-events: none;
        z-index: 1;
        animation: floatUp 4s linear forwards;
        --random-x: ${randomX};
    `;
    
    particle.style.left = Math.random() * window.innerWidth + 'px';
    particle.style.top = window.innerHeight + 'px';
    
    document.body.appendChild(particle);
    
    setTimeout(() => {
        particle.remove();
    }, 4000);
}

// Create particles periodically
setInterval(createFloatingParticle, 3000);

// Mobile menu toggle
const mobileMenuBtn = document.querySelector('.mobile-menu-btn');
const navLinks = document.querySelector('.nav-links');

mobileMenuBtn.addEventListener('click', () => {
    navLinks.style.display = navLinks.style.display === 'flex' ? 'none' : 'flex';
});

// Add some hover effects to feature cards
document.querySelectorAll('.feature-card').forEach(card => {
    card.addEventListener('mouseenter', function() {
        this.style.transform = 'translateY(-10px) scale(1.02)';
    });
    
    card.addEventListener('mouseleave', function() {
        this.style.transform = 'translateY(-5px) scale(1)';
    });
});

// Console welcome message
console.log(`
🚦 SignDetect AI - Traffic Sign Detection System
═══════════════════════════════════════════════

Welcome to the future of intelligent transportation!

✨ Features:
• 95%+ accuracy in real-time detection
• 120+ FPS processing speed
• 43 traffic sign categories
• Cross-platform deployment

🚀 Built with: TensorFlow, OpenCV, Python

Interested in contributing or learning more?
Visit: https://github.com/yourusername/sign-detection
`);

// Add some Easter eggs
let konamiCode = [];
const konamiSequence = [38, 38, 40, 40, 37, 39, 37, 39, 66, 65];

document.addEventListener('keydown', (e) => {
    konamiCode.push(e.keyCode);
    konamiCode = konamiCode.slice(-10);
    
    if (konamiCode.join(',') === konamiSequence.join(',')) {
        document.body.style.filter = 'hue-rotate(180deg)';
        setTimeout(() => {
            document.body.style.filter = 'none';
            alert('🎉 Easter egg activated! You found the secret code!');
        }, 2000);
    }
});
