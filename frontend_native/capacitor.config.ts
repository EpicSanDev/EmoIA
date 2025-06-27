import { CapacitorConfig } from '@capacitor/cli';

const config: CapacitorConfig = {
  appId: 'ai.emoia.app',
  appName: 'EmoIA',
  webDir: 'build',
  server: {
    androidScheme: 'https',
    // Configuration du tunnel pour se connecter au backend
    url: process.env.REACT_APP_TUNNEL_URL || 'https://emoia-ai.ngrok.io',
    cleartext: false
  },
  plugins: {
    SplashScreen: {
      launchShowDuration: 2000,
      launchAutoHide: true,
      backgroundColor: "#1976d2",
      androidSplashResourceName: "splash",
      androidScaleType: "CENTER_CROP",
      showSpinner: true,
      androidSpinnerStyle: "large",
      iosSpinnerStyle: "small",
      spinnerColor: "#ffffff"
    },
    StatusBar: {
      style: "DARK"
    },
    PushNotifications: {
      presentationOptions: ["badge", "sound", "alert"]
    },
    LocalNotifications: {
      smallIcon: "ic_stat_icon_config_sample",
      iconColor: "#488AFF",
      sound: "beep.wav"
    },
    Camera: {
      permissions: ["camera", "photos"]
    },
    Geolocation: {
      permissions: ["location"]
    },
    Device: {
      permissions: []
    },
    Network: {
      permissions: []
    },
    VoiceRecorder: {
      permissions: ["microphone"]
    }
  },
  android: {
    allowMixedContent: true,
    captureInput: true,
    webContentsDebuggingEnabled: true
  },
  ios: {
    scheme: "EmoIA",
    contentInset: "automatic"
  }
};

export default config;