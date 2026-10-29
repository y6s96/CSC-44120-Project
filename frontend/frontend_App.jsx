import React, { useState, useRef, useEffect } from 'react';

export default function App() {
  const [domain, setDomain] = useState('cars');
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [caption, setCaption] = useState('');
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [health, setHealth] = useState(null);
  const fileInputRef = useRef(null);

  const DOMAINS = {
    cars: { label: '🚗 Cars', hint: 'e.g., "The car is red"' },
    dogs: { label: '🐕 Dogs', hint: 'e.g., "The dog is brown"' },
    flowers: { label: '🌸 Flowers', hint: 'e.g., "The flower is pink"' },
    objects: { label: '⭕ Objects', hint: 'e.g., "The shape is a circle"' },
  };

  // Check model health
  useEffect(() => {
    fetch('/api/health')
      .then(r => r.json())
      .then(setHealth)
      .catch(e => console.error('Health check failed:', e));
  }, []);

  const handleImageChange = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (file.size > 5 * 1024 * 1024) {
      setError('Image must be < 5 MB');
      return;
    }

    if (!['image/png', 'image/jpeg'].includes(file.type)) {
      setError('Only PNG or JPEG allowed');
      return;
    }

    setImage(file);
    setError('');
    setPrediction(null);

    const reader = new FileReader();
    reader.onload = (e) => setPreview(e.target.result);
    reader.readAsDataURL(file);
  };

  const handlePredict = async (e) => {
    e.preventDefault();
    if (!image || !caption.trim()) {
      setError('Upload image and enter caption');
      return;
    }

    setLoading(true);
    setError('');
    setPrediction(null);

    try {
      const formData = new FormData();
      formData.append('domain', domain);
      formData.append('image', image);
      formData.append('caption', caption);

      const response = await fetch('/api/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.error || 'Prediction failed');
      }

      const result = await response.json();
      setPrediction(result);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const isModelReady = health?.model_ready;

  return (
    <div style={styles.container}>
      <header style={styles.header}>
        <div style={styles.headerContent}>
          <div>
            <h1 style={styles.title}>Cross-Modal Inconsistency Detector</h1>
            <p style={styles.subtitle}>MSc AI & Data Science · Hamza Javid</p>
          </div>
          <div style={styles.statusBox}>
            <div style={{ ...styles.statusIndicator, backgroundColor: isModelReady ? '#4caf50' : '#ff9800' }} />
            <span style={styles.statusText}>
              {isModelReady ? '✓ Ready' : '⏳ Loading...'}
            </span>
          </div>
        </div>
      </header>

      <main style={styles.main}>
        <form onSubmit={handlePredict} style={styles.form}>
          {/* Domain Selection */}
          <section style={styles.section}>
            <h2 style={styles.sectionTitle}>Step 1: Select Domain</h2>
            <div style={styles.domainGrid}>
              {Object.entries(DOMAINS).map(([key, val]) => (
                <button
                  key={key}
                  type="button"
                  onClick={() => setDomain(key)}
                  style={{
                    ...styles.domainButton,
                    backgroundColor: domain === key ? '#0f6b4a' : '#e0f2f1',
                    color: domain === key ? 'white' : '#0f3836',
                  }}
                >
                  {val.label}
                </button>
              ))}
            </div>
            <p style={styles.hint}>{DOMAINS[domain].hint}</p>
          </section>

          {/* Image Upload */}
          <section style={styles.section}>
            <h2 style={styles.sectionTitle}>Step 2: Upload Image</h2>
            <div style={styles.previewBox}>
              {preview ? (
                <img src={preview} alt="Preview" style={styles.previewImage} />
              ) : (
                <div style={styles.previewPlaceholder}>
                  <span>📷 No image selected</span>
                </div>
              )}
            </div>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/png,image/jpeg"
              onChange={handleImageChange}
              style={{ display: 'none' }}
            />
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              style={styles.uploadButton}
            >
              📤 Choose Image
            </button>
            {image && <p style={styles.filename}>✓ {image.name}</p>}
          </section>

          {/* Caption Input */}
          <section style={styles.section}>
            <h2 style={styles.sectionTitle}>Step 3: Write Caption</h2>
            <textarea
              value={caption}
              onChange={(e) => setCaption(e.target.value.slice(0, 180))}
              placeholder={`e.g., "${DOMAINS[domain].hint.replace('e.g., ', '').slice(0, -1)}"`}
              maxLength={180}
              style={styles.textarea}
              disabled={loading || !isModelReady}
            />
            <div style={styles.charCount}>
              {caption.length} / 180
            </div>
          </section>

          {/* Error Display */}
          {error && (
            <div style={styles.errorBox}>
              <strong>Error:</strong> {error}
            </div>
          )}

          {/* Predict Button */}
          <button
            type="submit"
            disabled={loading || !isModelReady || !image || !caption.trim()}
            style={{
              ...styles.predictButton,
              opacity: (loading || !isModelReady || !image || !caption.trim()) ? 0.6 : 1,
            }}
          >
            {loading ? '⏳ Analyzing...' : '🔍 Analyze Image & Caption'}
          </button>
        </form>

        {/* Result Display */}
        {prediction && (
          <section style={styles.resultSection}>
            <div
              style={{
                ...styles.resultBox,
                borderColor: prediction.prediction === 'consistent' ? '#4caf50' : '#ff6b6b',
                backgroundColor: prediction.prediction === 'consistent' ? '#f1f8e9' : '#ffebee',
              }}
            >
              <div style={styles.resultHeader}>
                <span style={styles.resultIcon}>
                  {prediction.prediction === 'consistent' ? '✅' : '❌'}
                </span>
                <div>
                  <h3 style={styles.resultTitle}>
                    {prediction.prediction === 'consistent' ? 'Consistent' : 'Inconsistent'}
                  </h3>
                  <p style={styles.resultText}>
                    Confidence: {(prediction.confidence * 100).toFixed(1)}%
                  </p>
                </div>
              </div>
              <hr style={styles.resultDivider} />
              <div style={styles.resultDetails}>
                <p><strong>Domain:</strong> {prediction.domain}</p>
                <p><strong>Caption:</strong> "{prediction.caption}"</p>
                <p><strong>Model Score:</strong> {prediction.score.toFixed(4)}</p>
              </div>
            </div>
          </section>
        )}
      </main>

      <footer style={styles.footer}>
        <p>CSC-44120 · Keele University · 2026</p>
        <p>Multi-Domain Image-Caption Inconsistency Detection</p>
      </footer>
    </div>
  );
}

const styles = {
  container: {
    fontFamily: 'Inter, -apple-system, sans-serif',
    backgroundColor: '#f8faf9',
    color: '#1a3d3d',
    minHeight: '100vh',
    display: 'flex',
    flexDirection: 'column',
  },
  header: {
    backgroundColor: '#0f3836',
    color: 'white',
    padding: '24px 32px',
    borderBottom: '1px solid #1a5a56',
  },
  headerContent: {
    maxWidth: '1200px',
    margin: '0 auto',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  title: {
    fontSize: '24px',
    fontWeight: 600,
    margin: '0 0 8px 0',
  },
  subtitle: {
    fontSize: '13px',
    opacity: 0.8,
    margin: 0,
  },
  statusBox: {
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
    backgroundColor: 'rgba(0,0,0,0.2)',
    padding: '8px 16px',
    borderRadius: '6px',
  },
  statusIndicator: {
    width: '8px',
    height: '8px',
    borderRadius: '50%',
  },
  statusText: {
    fontSize: '12px',
    fontWeight: 600,
  },
  main: {
    flex: 1,
    maxWidth: '800px',
    margin: '0 auto',
    padding: '40px 32px',
    width: '100%',
  },
  form: {
    display: 'flex',
    flexDirection: 'column',
    gap: '32px',
  },
  section: {
    backgroundColor: 'white',
    border: '1px solid #d4e0da',
    borderRadius: '12px',
    padding: '24px',
  },
  sectionTitle: {
    fontSize: '16px',
    fontWeight: 600,
    margin: '0 0 20px 0',
    color: '#0f3836',
  },
  domainGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(2, 1fr)',
    gap: '12px',
    marginBottom: '16px',
  },
  domainButton: {
    padding: '12px 16px',
    border: '1px solid #cfe0d9',
    borderRadius: '8px',
    fontSize: '14px',
    fontWeight: 600,
    cursor: 'pointer',
    transition: 'all 0.2s',
  },
  hint: {
    fontSize: '12px',
    color: '#7a8a7f',
    margin: '0',
  },
  previewBox: {
    backgroundColor: '#e8ede8',
    border: '1px solid #d4e0da',
    borderRadius: '9px',
    height: '280px',
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: '16px',
    overflow: 'hidden',
  },
  previewImage: {
    maxWidth: '100%',
    maxHeight: '100%',
    objectFit: 'contain',
  },
  previewPlaceholder: {
    fontSize: '16px',
    color: '#7a8a7f',
  },
  uploadButton: {
    width: '100%',
    padding: '12px',
    backgroundColor: 'white',
    border: '1px solid #cfe0d9',
    borderRadius: '6px',
    fontSize: '12px',
    fontWeight: 600,
    cursor: 'pointer',
    transition: 'background 0.2s',
  },
  filename: {
    fontSize: '11px',
    color: '#4caf50',
    margin: '8px 0 0 0',
  },
  textarea: {
    width: '100%',
    minHeight: '100px',
    padding: '12px',
    border: '1px solid #cfe0d9',
    borderRadius: '7px',
    backgroundColor: '#fcfdfc',
    color: '#1a3d3d',
    fontSize: '13px',
    fontFamily: 'inherit',
    resize: 'vertical',
    marginBottom: '8px',
  },
  charCount: {
    fontSize: '11px',
    color: '#7a8a7f',
    textAlign: 'right',
  },
  errorBox: {
    backgroundColor: '#ffebee',
    border: '1px solid #ef9a9a',
    borderRadius: '8px',
    padding: '16px',
    color: '#c62828',
    fontSize: '13px',
  },
  predictButton: {
    padding: '14px 24px',
    backgroundColor: '#0f6b4a',
    color: 'white',
    border: 'none',
    borderRadius: '7px',
    fontSize: '14px',
    fontWeight: 600,
    cursor: 'pointer',
    transition: 'background 0.2s',
  },
  resultSection: {
    marginTop: '24px',
  },
  resultBox: {
    border: '2px solid',
    borderRadius: '12px',
    padding: '24px',
  },
  resultHeader: {
    display: 'flex',
    gap: '16px',
    alignItems: 'flex-start',
  },
  resultIcon: {
    fontSize: '32px',
  },
  resultTitle: {
    fontSize: '20px',
    fontWeight: 600,
    margin: '0 0 4px 0',
  },
  resultText: {
    fontSize: '12px',
    margin: '0',
    opacity: 0.8,
  },
  resultDivider: {
    border: 'none',
    borderTop: '1px solid currentColor',
    opacity: 0.2,
    margin: '16px 0',
  },
  resultDetails: {
    fontSize: '12px',
  },
  footer: {
    borderTop: '1px solid #d4e0da',
    padding: '20px 32px',
    textAlign: 'center',
    color: '#7a8a7f',
    fontSize: '11px',
  },
};
