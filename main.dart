import 'package:flutter/material.dart';
import 'dart:convert';
import 'package:http/http.dart' as http;

void main() {
  runApp(const LearningBuddyApp());
}

class LearningBuddyApp extends StatelessWidget {
  const LearningBuddyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Friendly Learning Buddy',
      theme: ThemeData(
        primarySwatch: Colors.amber,
        fontFamily: 'Comic Sans MS',
        scaffoldBackgroundColor: const Color(0xFFFFECB3), // Warm, friendly background
      ),
      home: const ChatScreen(),
    );
  }
}

class ChatScreen extends StatefulWidget {
  const ChatScreen({super.key});

  @override
  State<ChatScreen> createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> {
  final TextEditingController _controller = TextEditingController();
  final List<ChatMessage> _messages = [];
  bool _isLoading = false;
  String _statusText = "Ready to chat!";
  Color _statusColor = const Color(0xFF4CAF50);
  
  // API Key for Google Gemini
  final String apiKey = "AIzaSyCcXz01pVbNOigauDYzanWePYU9a8ot9B0"; // Replace with your actual API key
  
  // System instruction for the learning buddy
  final String systemInstruction = """
You are Buddy, a friendly learning assistant for primary school children ages 6-12. Your personality:
- Enthusiastic and encouraging
- Patient and kind
- Speaks simply using age-appropriate vocabulary
- Explains concepts using examples kids can relate to
- Uses a warm, friendly tone
- Occasionally uses fun emojis
- Keeps answers short (2-4 sentences for younger kids)
- Avoids complex terminology without explanation
- Never uses inappropriate language or topics

When explaining concepts:
- Break down complex ideas into simple parts
- Use comparisons to everyday things kids understand
- Ask occasional questions to check understanding
- Praise effort and curiosity
- Encourage further exploration

Topics to focus on:
- Basic science, math, language arts, history, and geography
- Life skills and social-emotional learning
- Study strategies and learning tips
- Fun facts and interesting information
- Positive character development

Always maintain a positive, encouraging tone and make learning fun!
""";

  bool _hasInitialized = false;

  @override
  void initState() {
    super.initState();
    // Add welcome message
    _messages.add(
      ChatMessage(
        text: "Hi there! I'm your Learning Buddy! 🤖 Ask me anything and I'll help you learn. What would you like to know about today?",
        isUser: false,
      ),
    );
    
    // Initialize the chat
    _initializeChat();
  }
  
  Future<void> _initializeChat() async {
    if (_hasInitialized) return;
    
    try {
      // Send system instruction to initialize the chat
      await _sendRequestToGemini(systemInstruction, addToChat: false);
      _hasInitialized = true;
    } catch (e) {
      print("Error initializing chat: $e");
    }
  }

  void _handleSubmitted(String text) async {
    if (text.trim().isEmpty) return;
    
    _controller.clear();
    
    setState(() {
      _messages.add(ChatMessage(text: text, isUser: true));
      _isLoading = true;
      _statusText = "🤔 Thinking...";
      _statusColor = const Color(0xFFFF6B6B);
    });
    
    try {
      final response = await _sendRequestToGemini(text);
      
      setState(() {
        _messages.add(ChatMessage(text: response, isUser: false));
        _isLoading = false;
        _statusText = "Ready to chat!";
        _statusColor = const Color(0xFF4CAF50);
      });
    } catch (e) {
      setState(() {
        _isLoading = false;
        _statusText = "Error occurred! Try again.";
        _statusColor = Colors.red;
      });
      
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text("Oops! Something went wrong. Please try again!")),
      );
      
      print("Error: $e");
    }
  }

  Future<String> _sendRequestToGemini(String message, {bool addToChat = true}) async {
    final url = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=$apiKey';
    
    // Prepare the request body
    final body = jsonEncode({
      "contents": [
        {
          "role": "user",
          "parts": [{"text": message}]
        }
      ],
      "generation_config": {
        "temperature": 0.7,
        "top_k": 40,
        "top_p": 0.95,
        "max_output_tokens": 1024
      }
    });
    
    final response = await http.post(
      Uri.parse(url),
      headers: {
        'Content-Type': 'application/json',
      },
      body: body,
    );
    
    if (response.statusCode == 200) {
      final data = jsonDecode(response.body);
      return data['candidates'][0]['content']['parts'][0]['text'];
    } else {
      throw Exception('Failed to connect to Gemini API: ${response.statusCode}');
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text(
          'My Learning Buddy',
          style: TextStyle(
            fontFamily: 'Comic Sans MS',
            fontSize: 24,
            fontWeight: FontWeight.bold,
            color: Color(0xFF5D4037),
          ),
        ),
        backgroundColor: const Color(0xFFFFD54F),
        elevation: 0,
      ),
      body: Column(
        children: [
          Expanded(
            child: Container(
              color: const Color(0xFFFFF8E1),
              padding: const EdgeInsets.all(10),
              child: ListView.builder(
                itemCount: _messages.length,
                itemBuilder: (context, index) {
                  return _messages[index];
                },
              ),
            ),
          ),
          Container(
            padding: const EdgeInsets.symmetric(vertical: 5),
            color: const Color(0xFFFFECB3),
            child: Center(
              child: Text(
                _statusText,
                style: TextStyle(
                  fontFamily: 'Comic Sans MS',
                  fontSize: 10,
                  color: _statusColor,
                ),
              ),
            ),
          ),
          Container(
            color: const Color(0xFFFFD54F),
            padding: const EdgeInsets.symmetric(horizontal: 8.0, vertical: 10.0),
            child: Row(
              children: [
                Expanded(
                  child: TextField(
                    controller: _controller,
                    decoration: InputDecoration(
                      hintText: 'Type your question here...',
                      hintStyle: const TextStyle(color: Colors.grey),
                      contentPadding: const EdgeInsets.all(10.0),
                      border: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(12.0),
                        borderSide: BorderSide.none,
                      ),
                      filled: true,
                      fillColor: Colors.white,
                    ),
                    style: const TextStyle(
                      fontFamily: 'Comic Sans MS', 
                      fontSize: 14,
                    ),
                    onSubmitted: _handleSubmitted,
                  ),
                ),
                const SizedBox(width: 8.0),
                ElevatedButton(
                  onPressed: _isLoading ? null : () => _handleSubmitted(_controller.text),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: const Color(0xFFFF9800),
                    foregroundColor: Colors.white,
                    padding: const EdgeInsets.symmetric(horizontal: 15, vertical: 12),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(10),
                    ),
                  ),
                  child: Text(
                    'Ask!',
                    style: TextStyle(
                      fontFamily: 'Comic Sans MS',
                      fontSize: 14,
                      fontWeight: FontWeight.bold,
                      color: _isLoading ? Colors.grey : Colors.white,
                    ),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class ChatMessage extends StatelessWidget {
  final String text;
  final bool isUser;

  const ChatMessage({super.key, required this.text, required this.isUser});

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.symmetric(vertical: 10.0),
      child: Row(
        mainAxisAlignment: isUser ? MainAxisAlignment.end : MainAxisAlignment.start,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          if (!isUser) ...[
            CircleAvatar(
              backgroundColor: const Color(0xFFA5D6A7),
              child: Text('B', style: const TextStyle(color: Colors.white)),
            ),
            const SizedBox(width: 8.0),
          ],
          Expanded(
            child: Container(
              padding: const EdgeInsets.all(12.0),
              decoration: BoxDecoration(
                color: isUser ? const Color(0xFF90CAF9) : const Color(0xFFA5D6A7),
                borderRadius: BorderRadius.circular(8.0),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    isUser ? 'You' : 'Buddy',
                    style: const TextStyle(
                      fontWeight: FontWeight.bold,
                      fontSize: 12,
                    ),
                  ),
                  const SizedBox(height: 4.0),
                  Text(
                    text,
                    style: const TextStyle(
                      fontFamily: 'Comic Sans MS',
                      fontSize: 14,
                    ),
                  ),
                ],
              ),
            ),
          ),
          if (isUser) ...[
            const SizedBox(width: 8.0),
            CircleAvatar(
              backgroundColor: const Color(0xFF90CAF9),
              child: Text('Y', style: const TextStyle(color: Colors.white)),
            ),
          ],
        ],
      ),
    );
  }
}