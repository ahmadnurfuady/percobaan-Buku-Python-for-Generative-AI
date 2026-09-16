# 2.1 Lists for Conversation Histories and Batches
conversation = []
conversation.append({"role": "user", "content": "Hello"})
conversation.append({"role": "assistant", "content": "Hi! How can I help?"})
conversation.append({"role": "user", "content": "Explain RAG."})

print("Turns count:", len(conversation))
print("First turn:", conversation[0])
print("Last turn:", conversation[-1])
print("Slice [1:3]:", conversation[1:3])

# Useful list methods
scores = [0.91, 0.76, 0.88, 0.65, 0.95]
scores.sort(reverse=True)
print("Sorted scores:", scores)

# Remove by value vs by index
scores.remove(0.76)
popped = scores.pop()
print(f"Popped: {popped}, Remaining: {scores}")
