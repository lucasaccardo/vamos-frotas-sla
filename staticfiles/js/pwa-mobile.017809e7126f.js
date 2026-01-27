/* ===================================
   VAMOS FROTAS - PWA MOBILE JAVASCRIPT
   Progressive Web App Mobile Functionality
   =================================== */

(function() {
    'use strict';

    // ===================================
    // 1. PWA INSTALLATION & SERVICE WORKER
    // ===================================

    // Check if browser supports service workers
    if ('serviceWorker' in navigator) {
        window.addEventListener('load', () => {
            // Register service worker (to be implemented later)
            // navigator.serviceWorker.register('/sw.js')
            //     .then(registration => console.log('SW registered:', registration))
            //     .catch(err => console.log('SW registration failed:', err));
        });
    }

    // Handle PWA installation prompt
    let deferredPrompt;
    window.addEventListener('beforeinstallprompt', (e) => {
        e.preventDefault();
        deferredPrompt = e;
        // Show install button if needed
        const installBtn = document.querySelector('.pwa-install-btn');
        if (installBtn) {
            installBtn.style.display = 'block';
            installBtn.addEventListener('click', () => {
                deferredPrompt.prompt();
                deferredPrompt.userChoice.then((choiceResult) => {
                    if (choiceResult.outcome === 'accepted') {
                        console.log('User accepted the install prompt');
                    }
                    deferredPrompt = null;
                });
            });
        }
    });

    // ===================================
    // 2. MOBILE TAB BAR NAVIGATION
    // ===================================

    function initMobileTabBar() {
        const tabItems = document.querySelectorAll('.mobile-tab-bar .tab-item');
        
        // Set active tab based on current URL
        const currentPath = window.location.pathname;
        tabItems.forEach(tab => {
            const href = tab.getAttribute('href');
            if (href && currentPath.includes(href)) {
                tab.classList.add('active');
            }
        });

        // Add click animation
        tabItems.forEach(tab => {
            tab.addEventListener('click', function(e) {
                // Remove active from all tabs
                tabItems.forEach(t => t.classList.remove('active'));
                // Add active to clicked tab
                this.classList.add('active');
                
                // Add haptic feedback animation
                this.classList.add('haptic-feedback');
                setTimeout(() => {
                    this.classList.remove('haptic-feedback');
                }, 300);
            });
        });
    }

    // ===================================
    // 3. HAPTIC FEEDBACK SIMULATION
    // ===================================

    function addHapticFeedback() {
        // Add to all buttons
        const buttons = document.querySelectorAll('.btn, button, .card, .module-card');
        buttons.forEach(button => {
            button.classList.add('ripple');
            
            button.addEventListener('touchstart', function(e) {
                // Add visual feedback
                this.style.transform = 'scale(0.95)';
                
                // Vibrate if supported (haptic feedback)
                if ('vibrate' in navigator) {
                    navigator.vibrate(10); // 10ms vibration
                }
            });

            button.addEventListener('touchend', function(e) {
                this.style.transform = '';
            });
        });

        // Add to table rows
        const tableRows = document.querySelectorAll('.table tbody tr');
        tableRows.forEach(row => {
            row.addEventListener('touchstart', function() {
                if ('vibrate' in navigator) {
                    navigator.vibrate(5);
                }
            });
        });
    }

    // ===================================
    // 4. SKELETON SCREEN MANAGEMENT
    // ===================================

    function showSkeletonScreen(containerId) {
        const container = document.getElementById(containerId);
        if (!container) return;

        const skeletonHTML = `
            <div class="loading-container">
                <div class="skeleton-screen skeleton-card">
                    <div class="skeleton-screen skeleton-text large"></div>
                    <div class="skeleton-screen skeleton-text"></div>
                    <div class="skeleton-screen skeleton-text small"></div>
                </div>
                <div class="skeleton-screen skeleton-card">
                    <div class="skeleton-screen skeleton-text large"></div>
                    <div class="skeleton-screen skeleton-text"></div>
                    <div class="skeleton-screen skeleton-text small"></div>
                </div>
                <div class="skeleton-screen skeleton-card">
                    <div class="skeleton-screen skeleton-text large"></div>
                    <div class="skeleton-screen skeleton-text"></div>
                    <div class="skeleton-screen skeleton-text small"></div>
                </div>
            </div>
        `;
        
        container.innerHTML = skeletonHTML;
    }

    function hideSkeletonScreen(containerId) {
        const container = document.getElementById(containerId);
        if (!container) return;
        
        // Remove skeleton and show content with fade-in
        setTimeout(() => {
            const skeletons = container.querySelectorAll('.skeleton-screen');
            skeletons.forEach(skeleton => {
                skeleton.style.opacity = '0';
                skeleton.style.transition = 'opacity 0.3s ease';
            });
            
            setTimeout(() => {
                container.querySelector('.loading-container')?.remove();
            }, 300);
        }, 100);
    }

    // ===================================
    // 5. LOADING OVERLAY
    // ===================================

    function showLoadingOverlay(text = 'Carregando...') {
        const overlay = document.createElement('div');
        overlay.className = 'loading-overlay';
        overlay.id = 'pwa-loading-overlay';
        overlay.innerHTML = `
            <div class="pwa-loading-spinner"></div>
            <div class="loading-text">${text}</div>
        `;
        document.body.appendChild(overlay);
    }

    function hideLoadingOverlay() {
        const overlay = document.getElementById('pwa-loading-overlay');
        if (overlay) {
            overlay.style.opacity = '0';
            setTimeout(() => overlay.remove(), 300);
        }
    }

    // ===================================
    // 6. PREVENT HORIZONTAL SCROLL
    // ===================================

    function preventHorizontalScroll() {
        // Disable horizontal scroll globally
        document.body.style.overflowX = 'hidden';
        document.documentElement.style.overflowX = 'hidden';

        // Monitor for elements that might cause horizontal scroll
        const observer = new ResizeObserver(entries => {
            entries.forEach(entry => {
                const element = entry.target;
                if (element.scrollWidth > window.innerWidth) {
                    console.warn('Element causing horizontal scroll:', element);
                }
            });
        });

        // Observe body
        observer.observe(document.body);
    }

    // ===================================
    // 7. MOBILE VIEWPORT HEIGHT FIX
    // ===================================

    function fixMobileViewportHeight() {
        // Fix for mobile browsers where 100vh includes address bar
        const vh = window.innerHeight * 0.01;
        document.documentElement.style.setProperty('--vh', `${vh}px`);

        window.addEventListener('resize', () => {
            const vh = window.innerHeight * 0.01;
            document.documentElement.style.setProperty('--vh', `${vh}px`);
        });
    }

    // ===================================
    // 8. SMOOTH SCROLL ENHANCEMENT
    // ===================================

    function enhanceSmoothScroll() {
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function (e) {
                const href = this.getAttribute('href');
                if (href && href !== '#') {
                    e.preventDefault();
                    const target = document.querySelector(href);
                    if (target) {
                        target.scrollIntoView({
                            behavior: 'smooth',
                            block: 'start'
                        });
                    }
                }
            });
        });
    }

    // ===================================
    // 9. NETWORK STATUS MONITORING
    // ===================================

    function monitorNetworkStatus() {
        function updateOnlineStatus() {
            const isOnline = navigator.onLine;
            if (!isOnline) {
                showOfflineNotification();
            } else {
                hideOfflineNotification();
            }
        }

        function showOfflineNotification() {
            let notification = document.getElementById('offline-notification');
            if (!notification) {
                notification = document.createElement('div');
                notification.id = 'offline-notification';
                notification.className = 'alert alert-warning';
                notification.style.cssText = 'position: fixed; top: 20px; left: 50%; transform: translateX(-50%); z-index: 10000; min-width: 280px; text-align: center;';
                notification.innerHTML = '<i class="bi bi-wifi-off me-2"></i>Você está offline';
                document.body.appendChild(notification);
            }
        }

        function hideOfflineNotification() {
            const notification = document.getElementById('offline-notification');
            if (notification) {
                notification.remove();
            }
        }

        window.addEventListener('online', updateOnlineStatus);
        window.addEventListener('offline', updateOnlineStatus);
        updateOnlineStatus();
    }

    // ===================================
    // 10. PULL TO REFRESH (OPTIONAL)
    // ===================================

    function initPullToRefresh() {
        let startY = 0;
        let pullThreshold = 80;
        let isPulling = false;

        if (window.matchMedia('(max-width: 768px)').matches) {
            document.addEventListener('touchstart', (e) => {
                if (window.scrollY === 0) {
                    startY = e.touches[0].pageY;
                    isPulling = true;
                }
            });

            document.addEventListener('touchmove', (e) => {
                if (!isPulling) return;
                
                const currentY = e.touches[0].pageY;
                const pullDistance = currentY - startY;

                if (pullDistance > 0 && window.scrollY === 0) {
                    // Add visual feedback for pull to refresh
                    // This can be enhanced with a custom UI element
                }
            });

            document.addEventListener('touchend', (e) => {
                if (!isPulling) return;
                
                const currentY = e.changedTouches[0].pageY;
                const pullDistance = currentY - startY;

                if (pullDistance > pullThreshold && window.scrollY === 0) {
                    // Trigger refresh
                    location.reload();
                }

                isPulling = false;
            });
        }
    }

    // ===================================
    // 11. INITIALIZE ALL FEATURES
    // ===================================

    function initPWA() {
        console.log('Initializing PWA features...');
        
        // Wait for DOM to be fully loaded
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', init);
        } else {
            init();
        }

        function init() {
            initMobileTabBar();
            addHapticFeedback();
            preventHorizontalScroll();
            fixMobileViewportHeight();
            enhanceSmoothScroll();
            monitorNetworkStatus();
            // initPullToRefresh(); // Optional, can be enabled if needed
            
            console.log('PWA features initialized successfully');
        }
    }

    // ===================================
    // EXPORT FUNCTIONS FOR GLOBAL USE
    // ===================================

    window.VamosPWA = {
        showSkeletonScreen,
        hideSkeletonScreen,
        showLoadingOverlay,
        hideLoadingOverlay,
        init: initPWA
    };

    // Auto-initialize
    initPWA();

})();
