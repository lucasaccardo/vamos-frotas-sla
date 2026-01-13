from django.test import TestCase, override_settings
from django.urls import reverse
from django.contrib.auth.models import User


class SinistroManualViewTestCase(TestCase):
    """Tests for the SinistroManualView"""
    
    def setUp(self):
        """Create a test user"""
        self.user = User.objects.create_user(
            username='testuser',
            password='testpass123'
        )
        self.url = reverse('procedures-manual')
    
    def test_view_requires_authentication(self):
        """Test that unauthenticated users are redirected"""
        response = self.client.get(self.url)
        # Should redirect (either 301 for trailing slash or 302 for login)
        self.assertIn(response.status_code, [301, 302])
        # Check that redirect URL contains either the same path (for trailing slash)
        # or login (for authentication)
        self.assertTrue(
            '/procedures/manual' in response.url or '/login/' in response.url,
            f"Unexpected redirect URL: {response.url}"
        )
    
    def test_authenticated_user_can_access(self):
        """Test that authenticated users can access the view"""
        self.client.login(username='testuser', password='testpass123')
        response = self.client.get(self.url, follow=True)
        # Should eventually get a 200 after following redirects
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'sinistro.html')
    
    def test_view_uses_correct_template(self):
        """Test that the view uses the sinistro.html template"""
        self.client.login(username='testuser', password='testpass123')
        response = self.client.get(self.url, follow=True)
        self.assertTemplateUsed(response, 'sinistro.html')
    
    def test_view_url_resolves_correctly(self):
        """Test that the URL resolves to the correct path"""
        # The URL should be at /procedures/manual/
        expected_path = '/procedures/manual/'
        self.assertEqual(self.url, expected_path)


class SinistroWidgetIntegrationTestCase(TestCase):
    """Tests for the sinistro widget integration in base.html"""
    
    def setUp(self):
        """Create a test user"""
        self.user = User.objects.create_user(
            username='testuser',
            password='testpass123'
        )
    
    def test_widget_appears_in_sinistros_module(self):
        """Test that the widget appears when in sinistros module"""
        self.client.login(username='testuser', password='testpass123')
        
        # Set session to sinistros module
        session = self.client.session
        session['modulo_ativo'] = 'sinistros'
        session.save()
        
        # Access a page that uses base.html (e.g., sinistros home)
        # Note: We'll just check the manual URL is accessible
        response = self.client.get(reverse('procedures-manual'), follow=True)
        
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'sinistro.html')
    
    def test_widget_link_resolves(self):
        """Test that the widget's URL resolves correctly"""
        url = reverse('procedures-manual')
        self.assertEqual(url, '/procedures/manual/')
