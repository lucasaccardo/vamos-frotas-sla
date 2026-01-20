from django.test import TestCase, Client
from django.contrib.auth.models import User
from django.urls import reverse


class ManualSinistroTestCase(TestCase):
    """Test cases for Manual de Sinistro functionality"""
    
    def setUp(self):
        """Set up test client and user"""
        self.client = Client()
        self.user = User.objects.create_user(
            username='testuser',
            password='testpass123',
            email='test@example.com'
        )
    
    def test_manual_sinistro_url_exists(self):
        """Test that the manual_sinistro URL is configured correctly"""
        url = reverse('manual_sinistro')
        self.assertEqual(url, '/manual-sinistro/')
    
    def test_manual_sinistro_requires_login(self):
        """Test that manual_sinistro requires authentication"""
        url = reverse('manual_sinistro')
        response = self.client.get(url, follow=True)
        # Check that we were redirected to login page
        # The redirect chain will contain the login URL
        redirect_urls = [url for url, status in response.redirect_chain]
        self.assertTrue(any('login' in url.lower() for url in redirect_urls),
                       f"Expected redirect to login page, got: {redirect_urls}")
    
    def test_manual_sinistro_accessible_when_logged_in(self):
        """Test that authenticated users can access manual_sinistro"""
        self.client.login(username='testuser', password='testpass123')
        url = reverse('manual_sinistro')
        response = self.client.get(url, follow=True)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'vamos/manual_sinistro.html')
    
    def test_manual_sinistro_template_contains_expected_content(self):
        """Test that the manual template contains expected elements"""
        self.client.login(username='testuser', password='testpass123')
        url = reverse('manual_sinistro')
        response = self.client.get(url, follow=True)
        self.assertContains(response, 'Manual de Sinistro')
        self.assertContains(response, 'React')  # Template uses React
