# Migration Chatbot Scripts - Which One to Use?

## Quick Decision Guide

### 🎯 Just want to test if embeddings work?
→ Use **retrieval_no_llm_migration.py**
- No API key needed
- Shows raw retrieved documents
- Fast and simple

### 💬 Want a full conversational chatbot?
→ Use **retrieval_with_llm_migration.py** or **retrieval_with_llm_migration_enhanced.py**
- Requires Gemini API key
- Natural language answers
- Better user experience

### 🔬 Want to analyze retrieval quality?
→ Use **query_migration_rag.py**
- Command-line interface
- See exactly what's retrieved
- Good for debugging

## Detailed Comparison

### 1. retrieval_no_llm_migration.py
**Best for:** Testing, debugging, no API costs

✅ Pros:
- No API key required
- Fast responses (no LLM call)
- See exact retrieved documents
- Great for testing embedding quality
- Free to run unlimited queries

❌ Cons:
- Raw data output (not conversational)
- User must interpret results
- No answer synthesis

**Use when:**
- Testing your FAISS index
- Don't have/want to use API key
- Need to see exact source documents
- Debugging retrieval quality

---

### 2. retrieval_with_llm_migration.py
**Best for:** Simple chatbot deployment

✅ Pros:
- Clean, simple interface
- Natural language answers
- Easy to understand code
- Good starting point

❌ Cons:
- Requires Gemini API key
- Basic UI
- Costs per query (minimal)

**Use when:**
- Want conversational answers
- Okay with API costs
- Need simple deployment
- Don't need advanced features

**Requirements:**
```bash
pip install python-dotenv google-genai gradio
```

**.env file:**
```
GEMINI_API_KEY=your_key_here
```

---

### 3. retrieval_with_llm_migration_enhanced.py
**Best for:** Production deployment with features

✅ Pros:
- Beautiful tabbed UI
- Adjustable number of sources
- Shows which namespaces used
- Example questions built-in
- About page with instructions
- Copy button for answers
- Better formatted responses

❌ Cons:
- Requires Gemini API key
- Slightly more complex code
- Costs per query (minimal)

**Use when:**
- Deploying for end users
- Want professional UI
- Need transparency (shows sources)
- Want best user experience

**Requirements:**
Same as retrieval_with_llm_migration.py

---

### 4. query_migration_rag.py
**Best for:** Development and testing

✅ Pros:
- No UI dependencies
- Shows retrieval details
- Interactive terminal mode
- Good for debugging
- Fast iteration

❌ Cons:
- Terminal-only interface
- Not user-friendly for non-technical users
- No answer synthesis (just retrieval)

**Use when:**
- Developing the system
- Testing different queries
- Debugging retrieval
- Quick command-line access

## Feature Comparison Table

| Feature | No LLM | Basic LLM | Enhanced LLM | Terminal |
|---------|--------|-----------|--------------|----------|
| API Key Required | ❌ | ✅ | ✅ | ❌ |
| Web Interface | ✅ | ✅ | ✅ | ❌ |
| Natural Answers | ❌ | ✅ | ✅ | ❌ |
| Shows Sources | ✅ | ❌ | ✅ | ✅ |
| Adjustable Results | ✅ | ❌ | ✅ | ✅ |
| Example Queries | ✅ | ✅ | ✅ | ✅ |
| About/Help Page | ❌ | ❌ | ✅ | ❌ |
| Cost per Query | Free | ~$0.0001 | ~$0.0001 | Free |
| Best for | Testing | Simple Bot | Production | Debug |

## My Recommendation

### For Your Use Case (Migration Tracking):

**Phase 1 - Testing (Do this first):**
1. Run `retrieval_no_llm_migration.py`
2. Test various queries
3. Verify it finds the right namespaces
4. No cost, fast feedback

**Phase 2 - Deployment:**
1. Get Gemini API key (free tier available)
2. Use `retrieval_with_llm_migration_enhanced.py`
3. Deploy for your team
4. Professional UI with all features

**Phase 3 - Maintenance:**
- Keep `query_migration_rag.py` for debugging
- Use when you need to inspect raw retrieval

## API Key Information

### Getting Gemini API Key
1. Go to https://aistudio.google.com/apikey
2. Create new API key (free tier: 15 queries/min)
3. Save in `.env` file

### Costs
- Gemini 2.5 Flash: ~$0.0001 per query
- Most queries: <$0.01 per day
- Free tier: 1500 queries/day

## Running the Scripts

### No API Key Needed:
```bash
python retrieval_no_llm_migration.py
```

### With API Key:
```bash
# 1. Create .env file
echo "GEMINI_API_KEY=your_key" > .env

# 2. Run enhanced version
python retrieval_with_llm_migration_enhanced.py

# 3. Open browser to http://localhost:7860
```

## Troubleshooting

### "FAISS index not found"
→ Run `create_embeddings_migration.py` first

### "GEMINI_API_KEY not set"
→ Create `.env` file with your API key

### "Module not found: gradio"
→ Run `pip install gradio`

### Port 7860 already in use
→ Change port in the script:
```python
demo.launch(server_name="0.0.0.0", server_port=7861)
```

## Next Steps

1. **Start simple**: Test with `retrieval_no_llm_migration.py`
2. **Add intelligence**: Move to `retrieval_with_llm_migration_enhanced.py`
3. **Share**: Deploy on your network for team access
4. **Monitor**: Track which queries work well
5. **Improve**: Refine prompts and chunking strategy

Need help choosing? Start with the no-LLM version to validate retrieval, then upgrade to enhanced LLM version for production use.
