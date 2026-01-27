/**
 * VAMOS FROTAS - Premium Dark Theme JavaScript
 * Sidebar Collapse/Expand Functionality
 */

(function() {
    'use strict';
    
    // Check if we're on a page with sidebar
    const sidebar = document.querySelector('.sidebar');
    if (!sidebar) return;
    
    // Create toggle button if it doesn't exist
    let toggleBtn = document.querySelector('.sidebar-toggle');
    if (!toggleBtn) {
        toggleBtn = document.createElement('div');
        toggleBtn.className = 'sidebar-toggle';
        toggleBtn.innerHTML = '<i class="bi bi-chevron-left"></i>';
        toggleBtn.setAttribute('title', 'Recolher/Expandir Menu');
        sidebar.appendChild(toggleBtn);
    }
    
    // Get or initialize sidebar state from localStorage
    const STORAGE_KEY = 'vamos-sidebar-collapsed';
    const isCollapsed = localStorage.getItem(STORAGE_KEY) === 'true';
    
    // Apply initial state
    if (isCollapsed) {
        sidebar.classList.add('collapsed');
    }
    
    // Toggle function
    function toggleSidebar() {
        const willBeCollapsed = !sidebar.classList.contains('collapsed');
        
        sidebar.classList.toggle('collapsed');
        
        // Save state to localStorage
        localStorage.setItem(STORAGE_KEY, willBeCollapsed.toString());
        
        // Dispatch custom event for other components that might need to know
        window.dispatchEvent(new CustomEvent('sidebar-toggle', {
            detail: { collapsed: willBeCollapsed }
        }));
    }
    
    // Add click event to toggle button
    toggleBtn.addEventListener('click', function(e) {
        e.preventDefault();
        e.stopPropagation();
        toggleSidebar();
    });
    
    // Mobile menu handling
    const mobileBreakpoint = 992;
    
    function handleMobileMenu() {
        if (window.innerWidth <= mobileBreakpoint) {
            // Create mobile overlay if it doesn't exist
            let overlay = document.querySelector('.mobile-overlay');
            if (!overlay) {
                overlay = document.createElement('div');
                overlay.className = 'mobile-overlay';
                document.body.appendChild(overlay);
                
                // Close sidebar when clicking overlay
                overlay.addEventListener('click', function() {
                    sidebar.classList.remove('show');
                    overlay.classList.remove('show');
                });
            }
            
            // Create mobile toggle button in header if needed
            let mobileToggle = document.querySelector('.mobile-menu-toggle');
            if (!mobileToggle) {
                mobileToggle = document.createElement('button');
                mobileToggle.className = 'btn btn-outline-light mobile-menu-toggle d-lg-none';
                mobileToggle.innerHTML = '<i class="bi bi-list"></i>';
                mobileToggle.style.cssText = 'position: fixed; top: 20px; left: 20px; z-index: 2001;';
                document.body.appendChild(mobileToggle);
                
                // Toggle mobile menu
                mobileToggle.addEventListener('click', function() {
                    sidebar.classList.toggle('show');
                    overlay.classList.toggle('show');
                });
            }
        }
    }
    
    // Initialize mobile menu on load and resize
    handleMobileMenu();
    window.addEventListener('resize', handleMobileMenu);
    
    // Smooth scroll for internal links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            const href = this.getAttribute('href');
            if (href !== '#' && href !== '') {
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
    
    // Add animation classes to elements when they come into view
    function animateOnScroll() {
        const elements = document.querySelectorAll('.card, .table, .btn-danger, .btn-primary');
        
        const observer = new IntersectionObserver((entries) => {
            entries.forEach((entry, index) => {
                if (entry.isIntersecting) {
                    setTimeout(() => {
                        entry.target.classList.add('fade-in');
                    }, index * 50); // Stagger animation
                    observer.unobserve(entry.target);
                }
            });
        }, {
            threshold: 0.1
        });
        
        elements.forEach(element => {
            if (!element.classList.contains('fade-in')) {
                observer.observe(element);
            }
        });
    }
    
    // Run animation observer
    if ('IntersectionObserver' in window) {
        animateOnScroll();
    }
    
    // Enhanced table hover effects
    const tableRows = document.querySelectorAll('.table tbody tr');
    tableRows.forEach(row => {
        row.addEventListener('mouseenter', function() {
            this.style.transition = 'all 0.25s ease';
        });
    });
    
    // Add tooltip for collapsed sidebar items
    function updateTooltips() {
        const navLinks = sidebar.querySelectorAll('.nav-link');
        const isCollapsed = sidebar.classList.contains('collapsed');
        
        navLinks.forEach(link => {
            const textSpan = link.querySelector('span');
            if (textSpan && isCollapsed) {
                link.setAttribute('title', textSpan.textContent.trim());
            } else {
                link.removeAttribute('title');
            }
        });
    }
    
    // Update tooltips on toggle
    window.addEventListener('sidebar-toggle', updateTooltips);
    updateTooltips();
    
    // Add loading state to buttons when clicked
    document.querySelectorAll('button[type="submit"], a.btn').forEach(btn => {
        btn.addEventListener('click', function() {
            if (!this.classList.contains('no-loading')) {
                const icon = this.querySelector('i');
                if (icon) {
                    icon.className = 'bi bi-hourglass-split';
                }
            }
        });
    });
    
    console.log('✨ Premium Dark Theme initialized successfully');
})();
