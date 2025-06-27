import React, { useState, useRef, useEffect } from 'react';
import { useTranslation } from 'react-i18next';

interface Props {
  onTranscript: (text: string) => void;
  onAudioData?: (blob: Blob) => void;
  language?: string;
}

const VoiceInput: React.FC<Props> = ({ onTranscript, onAudioData, language = 'fr-FR' }) => {
  const { t } = useTranslation();
  const [isRecording, setIsRecording] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [isListening, setIsListening] = useState(false);
  const [volume, setVolume] = useState(0);
  
  const recognitionRef = useRef<any>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const animationRef = useRef<number | null>(null);

  useEffect(() => {
    // Vérifier si l'API Web Speech est disponible
    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    
    if (SpeechRecognition) {
      const recognition = new SpeechRecognition();
      recognition.continuous = true;
      recognition.interimResults = true;
      recognition.lang = language;

      recognition.onresult = (event: any) => {
        let finalTranscript = '';
        let interimTranscript = '';

        for (let i = event.resultIndex; i < event.results.length; i++) {
          const transcript = event.results[i][0].transcript;
          if (event.results[i].isFinal) {
            finalTranscript += transcript + ' ';
          } else {
            interimTranscript += transcript;
          }
        }

        setTranscript(finalTranscript + interimTranscript);
        
        if (finalTranscript) {
          onTranscript(finalTranscript.trim());
        }
      };

      recognition.onerror = (event: any) => {
        console.error('Erreur de reconnaissance vocale:', event.error);
        setIsListening(false);
      };

      recognition.onend = () => {
        setIsListening(false);
      };

      recognitionRef.current = recognition;
    }

    return () => {
      if (recognitionRef.current) {
        recognitionRef.current.stop();
      }
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [language, onTranscript]);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      
      // Configuration de l'analyseur audio pour la visualisation
      const audioContext = new AudioContext();
      const source = audioContext.createMediaStreamSource(stream);
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 256;
      source.connect(analyser);
      analyserRef.current = analyser;

      // Démarrer l'animation du volume
      const updateVolume = () => {
        if (analyserRef.current) {
          const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount);
          analyserRef.current.getByteFrequencyData(dataArray);
          const average = dataArray.reduce((a, b) => a + b) / dataArray.length;
          setVolume(average / 255);
          animationRef.current = requestAnimationFrame(updateVolume);
        }
      };
      updateVolume();

      // Configuration de l'enregistrement
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        audioChunksRef.current.push(event.data);
      };

      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        if (onAudioData) {
          onAudioData(audioBlob);
        }
        
        // Nettoyer
        stream.getTracks().forEach(track => track.stop());
        if (animationRef.current) {
          cancelAnimationFrame(animationRef.current);
        }
        setVolume(0);
      };

      mediaRecorder.start();
      
      // Démarrer la reconnaissance vocale
      if (recognitionRef.current) {
        recognitionRef.current.start();
        setIsListening(true);
      }
      
      setIsRecording(true);
    } catch (error) {
      console.error('Erreur lors du démarrage de l\'enregistrement:', error);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
    
    if (recognitionRef.current && isListening) {
      recognitionRef.current.stop();
      setIsListening(false);
    }
  };

  const toggleRecording = () => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  };

  return (
    <div className="voice-input-container">
      <div className="voice-controls">
        <button
          className={`voice-button ${isRecording ? 'recording' : ''}`}
          onClick={toggleRecording}
          aria-label={isRecording ? t('stopRecording') : t('startRecording')}
        >
          <div className="microphone-icon">
            {isRecording ? (
              <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
                <rect x="6" y="6" width="12" height="12" rx="2" />
              </svg>
            ) : (
              <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
                <path d="M12 14a3 3 0 0 0 3-3V6a3 3 0 0 0-6 0v5a3 3 0 0 0 3 3z" />
                <path d="M19 11a1 1 0 0 0-2 0 5 5 0 0 1-10 0 1 1 0 0 0-2 0 7 7 0 0 0 6 6.92V20H8a1 1 0 0 0 0 2h8a1 1 0 0 0 0-2h-3v-2.08A7 7 0 0 0 19 11z" />
              </svg>
            )}
          </div>
          
          {isRecording && (
            <div className="recording-indicator">
              <span className="recording-dot"></span>
              <span>{t('recording')}</span>
            </div>
          )}
        </button>

        {isRecording && (
          <div className="volume-meter">
            <div 
              className="volume-bar"
              style={{ 
                height: `${volume * 100}%`,
                backgroundColor: volume > 0.7 ? '#ff4444' : volume > 0.4 ? '#ffaa00' : '#44ff44'
              }}
            />
          </div>
        )}
      </div>

      {transcript && (
        <div className="transcript-display">
          <p>{transcript}</p>
        </div>
      )}
    </div>
  );
};

export default VoiceInput;