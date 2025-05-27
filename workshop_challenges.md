# Workshop Challenges - GraphRAG and AI Agents

## Module 1: Graph Basics Challenges

### Challenge 1.1: Skill Network Explorer 🔍
**Difficulty: Beginner**
Write a Cypher query to find the "skill influencers" - people who know the most rare skills (skills known by fewer than 5 people).

**Bonus**: Create a visualisation showing the relationship between rare skills and the people who know them.

### Challenge 1.2: Community Detective 🕵️
**Difficulty: Intermediate**
The dataset has `leiden_community` properties. Write queries to:
1. Find which community has the most diverse skill set
2. Identify "bridge people" - individuals whose skills span multiple communities
3. Discover the most "specialized" community (highest skill overlap within the community)

### Challenge 1.3: Skill Similarity Network 🌐
**Difficulty: Advanced**
Create a new relationship type `SKILL_COOCCURRENCE` that connects skills frequently found together:
1. Calculate how often skills appear together on the same person
2. Create relationships for skill pairs that co-occur more than 3 times
3. Find the most "central" skills in this network

## Module 2: Agent Challenges 

### Challenge 3.1: Career Path Advisor 🚀
**Difficulty: Beginner**
Create a new agent tool that suggests career progression paths:
- Input: Current person's skills
- Output: Recommended skills to learn next based on similar successful people
- Include reasoning about why these skills are recommended

### Challenge 3.2: Team Formation Agent 👥
**Difficulty: Intermediate**
Build an agent that can form optimal teams:
1. **Tool**: `form_project_team(required_skills: List[str], team_size: int)`
2. Find people with complementary skills
3. Minimize skill overlap while maximizing coverage
4. Consider community diversity for better collaboration

## Bonus Creative Challenges

### Challenge 5.1: Skill Compatibility Score 💕
Create a "professional compatibility" system like a dating app but for work partnerships. Calculate compatibility scores based on complementary skills, similar experience levels, and community connections.

### Challenge 5.2: Knowledge Transfer Optimizer 🎓
Build an agent that identifies optimal knowledge transfer paths - who should teach what to whom to maximize organizational learning efficiency.

**Remember**: The goal is to learn and have fun! Don't hesitate to ask for help, collaborate with others, and experiment with different approaches. The best solutions often come from combining concepts in unexpected ways! 🚀 