import React, { useEffect, useState } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { 
  Container, 
  AppBar, 
  Toolbar, 
  Typography, 
  IconButton, 
  Badge,
  Fab,
  Box,
  Alert,
  Snackbar
} from '@mui/material';
import {
  Menu as MenuIcon,
  Notifications as NotificationsIcon,
  Mic as MicIcon,
  Psychology as PsychologyIcon
} from '@mui/icons-material';

// Capacitor imports pour les fonctionnalités natives
import { Capacitor } from '@capacitor/core';
import { StatusBar, Style } from '@capacitor/status-bar';
import { SplashScreen } from '@capacitor/splash-screen';
import { Network } from '@capacitor/network';
import { PushNotifications } from '@capacitor/push-notifications';

// Components personnalisés
import ChatInterface from './components/ChatInterface';
import EmotionalDashboard from './components/EmotionalDashboard';
import VoiceInput from './components/VoiceInput';
import SettingsPage from './components/SettingsPage';
import OfflineMode from './components/OfflineMode';
import TunnelConnectionManager from './services/TunnelConnectionManager';
import { useAppDispatch, useAppSelector } from './hooks/redux';
import { setNetworkStatus, setTunnelStatus } from './store/appSlice';

// Service Worker pour PWA
import { registerSW } from './services/serviceWorker';

// Styles et thème
import './App.css';

const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#f50057',
    },
    background: {
      default: '#f5f5f5',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
  },
  components: {
    MuiAppBar: {
      styleOverrides: {
        root: {
          backdropFilter: 'blur(20px)',
          backgroundColor: 'rgba(25, 118, 210, 0.8)',
        },
      },
    },
  },
});

const App: React.FC = () => {
  const dispatch = useAppDispatch();
  const { isOnline, tunnelConnected, notifications } = useAppSelector(state => state.app);
  
  const [isNative, setIsNative] = useState(false);
  const [showVoiceInput, setShowVoiceInput] = useState(false);
  const [alert, setAlert] = useState<{ type: 'success' | 'error' | 'warning', message: string } | null>(null);

  useEffect(() => {
    initializeApp();
  }, []);

  const initializeApp = async () => {
    // Vérifier si on est sur une plateforme native
    setIsNative(Capacitor.isNativePlatform());

    if (Capacitor.isNativePlatform()) {
      // Configuration pour les plateformes natives
      await StatusBar.setStyle({ style: Style.Dark });
      await SplashScreen.hide();
      
      // Initialiser les notifications push
      await initializePushNotifications();
    }

    // Enregistrer le Service Worker pour PWA
    registerSW();

    // Initialiser la gestion réseau
    initializeNetworkMonitoring();

    // Initialiser la connexion tunnel
    initializeTunnelConnection();
  };

  const initializePushNotifications = async () => {
    if (!Capacitor.isNativePlatform()) return;

    try {
      let permStatus = await PushNotifications.checkPermissions();

      if (permStatus.receive === 'prompt') {
        permStatus = await PushNotifications.requestPermissions();
      }

      if (permStatus.receive !== 'granted') {
        throw new Error('Permissions de notification refusées');
      }

      await PushNotifications.register();

      PushNotifications.addListener('registration', token => {
        console.log('Push registration success, token: ' + token.value);
      });

      PushNotifications.addListener('registrationError', err => {
        console.error('Registration error: ', err.error);
      });

      PushNotifications.addListener('pushNotificationReceived', notification => {
        console.log('Push notification received: ', notification);
        // Traiter la notification reçue
      });

      PushNotifications.addListener('pushNotificationActionPerformed', notification => {
        console.log('Push notification action performed', notification.actionId, notification.inputValue);
      });

    } catch (error) {
      console.error('Erreur initialisation notifications:', error);
    }
  };

  const initializeNetworkMonitoring = () => {
    Network.addListener('networkStatusChange', status => {
      dispatch(setNetworkStatus(status.connected));
      
      if (!status.connected) {
        setAlert({
          type: 'warning',
          message: 'Connexion perdue - Mode hors ligne activé'
        });
      } else {
        setAlert({
          type: 'success',
          message: 'Connexion rétablie'
        });
      }
    });

    // Vérifier le statut initial
    Network.getStatus().then(status => {
      dispatch(setNetworkStatus(status.connected));
    });
  };

  const initializeTunnelConnection = () => {
    const tunnelManager = new TunnelConnectionManager();
    
    tunnelManager.onStatusChange((connected: boolean) => {
      dispatch(setTunnelStatus(connected));
      
      if (connected) {
        setAlert({
          type: 'success',
          message: '🌐 Connecté au serveur EmoIA via tunnel sécurisé'
        });
      } else {
        setAlert({
          type: 'error',
          message: '🔴 Connexion au serveur perdue'
        });
      }
    });

    tunnelManager.connect();
  };

  const handleVoiceInputToggle = () => {
    setShowVoiceInput(!showVoiceInput);
  };

  const handleCloseAlert = () => {
    setAlert(null);
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <Box sx={{ flexGrow: 1 }}>
          {/* AppBar avec design natif */}
          <AppBar position="fixed" elevation={0}>
            <Toolbar>
              <IconButton
                size="large"
                edge="start"
                color="inherit"
                aria-label="menu"
                sx={{ mr: 2 }}
              >
                <MenuIcon />
              </IconButton>
              
              <PsychologyIcon sx={{ mr: 2 }} />
              <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
                EmoIA
              </Typography>

              {/* Indicateur de statut connexion */}
              <Box sx={{ display: 'flex', alignItems: 'center', mr: 2 }}>
                <Box
                  sx={{
                    width: 8,
                    height: 8,
                    borderRadius: '50%',
                    backgroundColor: tunnelConnected ? '#4caf50' : '#f44336',
                    mr: 1
                  }}
                />
                <Typography variant="caption">
                  {tunnelConnected ? 'En ligne' : 'Hors ligne'}
                </Typography>
              </Box>

              <IconButton size="large" color="inherit">
                <Badge badgeContent={notifications.length} color="secondary">
                  <NotificationsIcon />
                </Badge>
              </IconButton>
            </Toolbar>
          </AppBar>

          {/* Contenu principal */}
          <Container 
            maxWidth="md" 
            sx={{ 
              mt: 8, 
              mb: 8,
              px: { xs: 1, sm: 2 },
              minHeight: 'calc(100vh - 128px)'
            }}
          >
            <Routes>
              <Route 
                path="/" 
                element={
                  isOnline ? (
                    <ChatInterface />
                  ) : (
                    <OfflineMode />
                  )
                } 
              />
              <Route path="/dashboard" element={<EmotionalDashboard />} />
              <Route path="/settings" element={<SettingsPage />} />
            </Routes>
          </Container>

          {/* Bouton flottant pour l'entrée vocale */}
          <Fab
            color="primary"
            aria-label="voice input"
            sx={{
              position: 'fixed',
              bottom: 16,
              right: 16,
              zIndex: 1000
            }}
            onClick={handleVoiceInputToggle}
          >
            <MicIcon />
          </Fab>

          {/* Interface d'entrée vocale */}
          {showVoiceInput && (
            <VoiceInput
              open={showVoiceInput}
              onClose={() => setShowVoiceInput(false)}
            />
          )}

          {/* Alertes système */}
          <Snackbar
            open={!!alert}
            autoHideDuration={4000}
            onClose={handleCloseAlert}
            anchorOrigin={{ vertical: 'top', horizontal: 'center' }}
          >
            {alert && (
              <Alert
                onClose={handleCloseAlert}
                severity={alert.type}
                variant="filled"
                sx={{ width: '100%' }}
              >
                {alert.message}
              </Alert>
            )}
          </Snackbar>
        </Box>
      </Router>
    </ThemeProvider>
  );
};

export default App;