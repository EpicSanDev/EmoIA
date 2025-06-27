import React from 'react';
import { useTranslation } from 'react-i18next';
import ReactCountryFlag from 'react-country-flags';

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
        <ReactCountryFlag countryCode="FR" svg style={{ width: '1.5em', height: '1.5em' }} />
      </button>
      <button 
        onClick={() => changeLanguage('en')} 
        className={i18n.language === 'en' ? 'active' : ''}
        aria-label="English"
      >
        <ReactCountryFlag countryCode="GB" svg style={{ width: '1.5em', height: '1.5em' }} />
      </button>
      <button 
        onClick={() => changeLanguage('es')} 
        className={i18n.language === 'es' ? 'active' : ''}
        aria-label="Español"
      >
        <ReactCountryFlag countryCode="ES" svg style={{ width: '1.5em', height: '1.5em' }} />
      </button>
    </div>
  );
};

export default LanguageSwitcher;