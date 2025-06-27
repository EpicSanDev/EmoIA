import React from 'react';
import { useTranslation } from 'react-i18next';
import { FlagIcon } from 'react-country-flags';

const LanguageSwitcher: React.FC = () => {
  const { i18n } = useTranslation();

  const changeLanguage = (lng: string) => {
    i18n.changeLanguage(lng);
    localStorage.setItem('emoia-lang', lng);
  };

  return (
    <div className="lang-switcher">
      <button 
        onClick={() => changeLanguage('fr')} 
        className={i18n.language === 'fr' ? 'active' : ''}
        aria-label="Français"
      >
        <FlagIcon country="FR" />
      </button>
      <button 
        onClick={() => changeLanguage('en')} 
        className={i18n.language === 'en' ? 'active' : ''}
        aria-label="English"
      >
        <FlagIcon country="GB" />
      </button>
      <button 
        onClick={() => changeLanguage('es')} 
        className={i18n.language === 'es' ? 'active' : ''}
        aria-label="Español"
      >
        <FlagIcon country="ES" />
      </button>
    </div>
  );
};

export default LanguageSwitcher;