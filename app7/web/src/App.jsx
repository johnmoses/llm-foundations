import React, { useState } from 'react';
import { generateAnswer } from './api';

function App() {
  const [query, setQuery] = useState('');
  const [answer, setAnswer] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;
    setLoading(true);
    setError(null);
    setAnswer('');
    try {
      const data = await generateAnswer(query);
      setAnswer(data.answer);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: 600, margin: '2rem auto', fontFamily: 'Arial, sans-serif' }}>
      <h1>Finance Q&A Chatbot</h1>
      <form onSubmit={handleSubmit}>
        <textarea
          rows={5}
          style={{ width: '100%', fontSize: 16, padding: 10 }}
          placeholder="Ask a finance question..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          disabled={loading}
        />
        <button type="submit" disabled={loading} style={{ marginTop: 10, padding: '10px 20px', fontSize: 16 }}>
          {loading ? 'Thinking...' : 'Ask'}
        </button>
      </form>
      {error && <p style={{ color: 'red' }}>Error: {error}</p>}
      {answer && (
        <div style={{ marginTop: 20, whiteSpace: 'pre-wrap', backgroundColor: '#f0f0f0', padding: 15, borderRadius: 5 }}>
          <strong>Answer:</strong>
          <p>{answer}</p>
        </div>
      )}
    </div>
  );
}

export default App;
