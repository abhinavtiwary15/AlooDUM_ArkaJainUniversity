import React, { useState, useEffect } from 'react';
import { ModelCardData, PatientSummary } from '../../shared/types';
import { checkApiHealth, fetchModelCard, fetchPatients } from './services/api';
import { Navbar } from './components/Navbar';
import { HomeView } from './components/HomeView';
import { TriageBoard } from './components/TriageBoard';
import { PatientTwinView } from './components/PatientTwinView';
import { EvidenceView } from './components/EvidenceView';

export type AppTab = 'home' | 'triage' | 'patient' | 'evidence';

export const App: React.FC = () => {
  const getTabFromHash = (): AppTab => {
    const hash = window.location.hash.replace('#', '').toLowerCase();
    if (hash === 'triage') return 'triage';
    if (hash === 'patient') return 'patient';
    if (hash === 'evidence') return 'evidence';
    return 'home';
  };

  const [currentTab, setCurrentTab] = useState<AppTab>(getTabFromHash);
  const [patients, setPatients] = useState<PatientSummary[]>([]);
  const [selectedPatientId, setSelectedPatientId] = useState<string>('2066');
  const [modelCard, setModelCard] = useState<ModelCardData | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [showRecordedOutcome, setShowRecordedOutcome] = useState<boolean>(false);

  // Sync hash on navigation
  useEffect(() => {
    const handleHashChange = () => {
      setCurrentTab(getTabFromHash());
    };
    window.addEventListener('hashchange', handleHashChange);
    return () => window.removeEventListener('hashchange', handleHashChange);
  }, []);

  const handleSelectTab = (tab: AppTab) => {
    setCurrentTab(tab);
    window.location.hash = tab === 'home' ? '' : tab;
    window.scrollTo({ top: 0, behavior: 'auto' });
  };

  const handleSelectPatient = (patientId: string) => {
    setSelectedPatientId(patientId);
    handleSelectTab('patient');
  };

  // Initial load
  useEffect(() => {
    async function init() {
      setIsLoading(true);
      try {
        await checkApiHealth();
        const [pts, mc] = await Promise.all([fetchPatients(), fetchModelCard()]);
        setPatients(pts);
        setModelCard(mc);
        if (pts.length > 0 && !pts.some((p) => p.patientId === selectedPatientId)) {
          setSelectedPatientId(pts[0].patientId);
        }
      } catch (err) {
        console.error('Initialization error:', err);
      } finally {
        setIsLoading(false);
      }
    }
    init();
  }, []);

  const selectedPatient = patients.find((p) => p.patientId === selectedPatientId) || patients[0] || null;

  return (
    <div className="min-h-screen bg-white text-[#0E1330] flex flex-col font-sans">
      <Navbar
        currentTab={currentTab}
        onSelectTab={handleSelectTab}
        selectedPatientId={selectedPatientId}
      />

      <main className="flex-1 w-full">
        {isLoading ? (
          <div className="flex flex-col items-center justify-center py-32 font-mono text-sm text-[#4A4F73]">
            <span>Loading...</span>
          </div>
        ) : (
          <>
            {currentTab === 'home' && (
              <HomeView
                onOpenTriage={() => handleSelectTab('triage')}
                onOpenEvidence={() => handleSelectTab('evidence')}
                onSelectPatient={handleSelectPatient}
                modelCard={modelCard}
                patients={patients}
              />
            )}

            {currentTab === 'triage' && (
              <TriageBoard
                patients={patients}
                selectedPatientId={selectedPatientId}
                onSelectPatient={handleSelectPatient}
                modelCard={modelCard}
                showRecordedOutcome={showRecordedOutcome}
                onToggleRecordedOutcome={setShowRecordedOutcome}
              />
            )}

            {currentTab === 'patient' && (
              <PatientTwinView
                patient={selectedPatient}
                onBackToTriage={() => handleSelectTab('triage')}
                modelCard={modelCard}
                showRecordedOutcome={showRecordedOutcome}
                onToggleRecordedOutcome={setShowRecordedOutcome}
              />
            )}

            {currentTab === 'evidence' && <EvidenceView modelCard={modelCard} />}
          </>
        )}
      </main>
    </div>
  );
};

export default App;
