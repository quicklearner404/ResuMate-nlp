import json
import random
from datetime import datetime, timedelta

def generate_tech_focused_resumes(num_samples=2000):
    """Generate tech-focused synthetic resume data with PROPER NER annotations"""
    
    # KEEPING ALL ORIGINAL EXTENSIVE DATA
    pakistani_names = [
        # Male names
        "Ahmed Khan", "Mohammad Hassan", "Bilal Ahmed", "Usman Khan", "Omar Farooq",
        "Haris Mahmood", "Tariq Javed", "Kamran Butt", "Faisal Akhtar", "Shahid Parvez",
        "Imran Yousuf", "Nasir Abbas", "Asif Nadeem", "Waqar Ahmed", "Kashif Raza",
        "Zubair Malik", "Saad Ali", "Rizwan Cheema", "Noman Akhtar", "Adnan Sheikh",
        "Salman Raza", "Farhan Iqbal", "Junaid Ahmed", "Abdullah Khan", "Hamza Yousuf",
        "Ali Raza", "Zain ul Abideen", "Fahad Malik", "Arsalan Khan", "Shoaib Akhtar",
        "Waseem Akram", "Babar Azam", "Hasan Ali", "Shadab Khan", "Fakhar Zaman",
        
        # Female names  
        "Fatima Ali", "Ayesha Raza", "Sana Malik", "Zainab Shah", "Hina Iqbal",
        "Sadia Rahman", "Nadia Chaudhry", "Amina Sheikh", "Rabia Hussain", "Saima Akram",
        "Sara Qureshi", "Mehwish Aslam", "Zara Khan", "Saba Mirza", "Noreen Malik",
        "Ayesha Siddiqui", "Maryam Khan", "Hira Ahmed", "Sadia Khan", "Bushra Raza",
        "Anum Fatima", "Sanaullah Khan", "Zoya Malik", "Aisha Sheikh", "Rukhsar Ahmed",
        "Kiran Khan", "Sahar Ali", "Mahnoor Butt", "Rida Javed", "Sundas Malik"
    ]
    
    western_names = [
        "John Smith", "Sarah Johnson", "Michael Brown", "Emily Davis", "David Wilson",
        "Jennifer Miller", "Christopher Moore", "Jessica Taylor", "Matthew Anderson",
        "Amanda Thomas", "Daniel Jackson", "Ashley White", "James Harris", "Elizabeth Martin",
        "Robert Garcia", "Michelle Rodriguez", "William Martinez", "Laura Hernandez", 
        "Joseph Lopez", "Stephanie Gonzalez", "Charles Perez", "Rebecca Torres", 
        "Thomas Sanchez", "Melissa Ramirez", "Richard Flores", "Kevin Chen", "Lisa Wang",
        "Brian Kim", "Nancy Patel", "Steven Nguyen", "Amy Gupta", "Eric Sharma", "Karen Zhang",
        "Mark Thompson", "Lisa Anderson", "Paul Wilson", "Karen Davis", "Jason Brown",
        "Nicole Miller", "Andrew Taylor", "Heather Moore", "Joshua Martin", "Stephanie Lee"
    ]
    
    all_names = pakistani_names + western_names
    
    # KEEPING ALL ORIGINAL COMPANIES
    pakistani_tech_companies = [
        "Systems Limited", "TRG Pakistan", "Avanza Solutions", "Netsol Technologies",
        "10Pearls", "Techlogix", "LMKR", "Softech", "Contour Software", "Tkxel",
        "Arpatech", "Folio3", "Sastaticket.pk", "Careem Pakistan", "Foodpanda Pakistan",
        "Daraz Pakistan", "Bykea", "Airlift", "Finja", "SadaPay", "QisstPay",
        "PakWheels", "Zameen.com", "Rozee.pk", "ApnaHR", "Trukkr", "BridgeLinx",
        "PriceOye", "Telemart", "Shopsy", "DigiKhata", "Kistpay", "CreditBook",
        "Telenor Pakistan", "Jazz", "PTCL", "Ufone", "Zong", "Mobilink",
        "Wateen Telecom", "Cybernet", "Nayatel", "Optix",
        "HBL Digital", "UBL Digital", "MCB Digital", "Bank Alfalah Digital",
        "JS Bank Digital", "Meezan Bank Digital", "Habib Bank Digital",
        "Ovex Technologies", "Elixir Technologies", "Devsinc", "Devsquad",
        "Panacloud", "Rolustech", "Xord", "Blockchain Solutions Pakistan",
        "Goto.com.pk", "Homeshopping.pk", "Symbios.pk", "Mall91 Pakistan",
        "Cheetay.pk", "Clicky.pk", "Shopistan.pk"
    ]
    
    international_tech_companies = [
        "Google", "Microsoft", "Apple", "Amazon", "Meta", "Netflix", 
        "Tesla", "SpaceX", "Twitter", "LinkedIn", "Uber", "Airbnb",
        "IBM", "Oracle", "Salesforce", "Adobe", "SAP", "VMware", "Citrix",
        "Cisco", "Juniper Networks", "Palo Alto Networks", "Fortinet",
        "Intel", "AMD", "NVIDIA", "Qualcomm", "Dell Technologies", "HP Inc",
        "Lenovo", "Samsung Electronics", "Sony", "Panasonic",
        "Atlassian", "GitHub", "GitLab", "Slack", "Zoom", "Shopify", "Square",
        "Stripe", "PayPal", "Intuit", "Workday", "ServiceNow",
        "MongoDB", "Elastic", "Databricks", "Snowflake", "Splunk", "Tableau",
        "Palantir", "Datadog", "New Relic", "AppDynamics",
        "CrowdStrike", "Okta", "Cloudflare", "Akamai", "Fastly", "Zscaler",
        "CyberArk", "Tenable", "Rapid7", "Proofpoint"
    ]
    
    tech_service_companies = [
        "Accenture", "Capgemini", "Cognizant", "Infosys", "TCS", "Wipro", 
        "HCL Technologies", "Tech Mahindra", "Deloitte Digital", "EY Technology",
        "PwC Technology", "KPMG Tech", "IBM iX", "DXC Technology", "NTT Data",
        "Fujitsu", "NEC", "Hitachi", "Tata Consultancy Services", "Mindtree",
        "LTI", "Mphasis", "Hexaware", "Persistent Systems", "Cybage",
        "Globant", "Endava", "EPAM Systems", "Luxoft", "SoftServe",
        "Publicis Sapient", "Wipro Digital", "Infosys Digital", "Virtusa",
        "Syntel", "ITC Infotech", "Maveric Systems", "Sonata Software"
    ]
    
    all_companies = pakistani_tech_companies + international_tech_companies + tech_service_companies
    
    # KEEPING ALL ORIGINAL LOCATIONS
    pakistani_tech_hubs = [
        "Karachi", "Lahore", "Islamabad", "Rawalpindi", "Faisalabad",
        "Peshawar", "Multan", "Gujranwala", "Sialkot", "Hyderabad",
        "Karachi DHA", "Lahore DHA", "Islamabad F-7", "Rawalpindi Bahria Town",
        "Lahore Gulberg", "Karachi Clifton", "Islamabad G-11", "Lahore Model Town",
        "Karachi Korangi", "Lahore Ferozepur Road", "Islamabad I-9",
        "Karachi SITE Area", "Lahore Bund Road"
    ]
    
    international_tech_hubs = [
        "Silicon Valley", "San Francisco", "Seattle", "Austin", "Boston",
        "New York", "Los Angeles", "Chicago", "San Diego", "Denver",
        "Portland", "Raleigh", "Atlanta", "Miami", "Phoenix",
        "London", "Berlin", "Amsterdam", "Stockholm", "Zurich", "Paris",
        "Dublin", "Barcelona", "Munich", "Copenhagen", "Helsinki",
        "Singapore", "Bangalore", "Hyderabad", "Tokyo", "Seoul", "Shanghai",
        "Hong Kong", "Shenzhen", "Taipei", "Bangkok", "Kuala Lumpur",
        "Toronto", "Vancouver", "Sydney", "Melbourne", "Dubai", "Tel Aviv"
    ]
    
    all_locations = pakistani_tech_hubs + international_tech_hubs
    
    # KEEPING ALL ORIGINAL DEGREES
    tech_degrees = [
        "B.S. Computer Science", "B.S. Software Engineering", "B.S. Computer Engineering",
        "B.S. Information Technology", "B.S. Data Science", "B.S. Artificial Intelligence",
        "B.S. Cybersecurity", "B.S. Cloud Computing", "Bachelors in Computer Applications",
        "B.Tech Computer Science", "B.E. Computer Engineering", "B.Sc. IT",
        "B.S. Information Systems", "B.S. Computer Systems", "B.S. Network Engineering",
        "M.S. Computer Science", "M.S. Software Engineering", "M.S. Data Science",
        "M.S. Artificial Intelligence", "M.S. Cybersecurity", "M.S. Cloud Computing",
        "M.S. Information Technology", "M.S. Computer Engineering", "Masters in Machine Learning",
        "M.Tech Computer Science", "M.E. Software Engineering", "M.Sc. Data Analytics",
        "Ph.D. Computer Science", "Ph.D. Artificial Intelligence", "Ph.D. Data Science",
        "Executive MBA Technology Management", "Postgraduate Diploma in Cloud Computing",
        "Advanced Certificate in Cybersecurity", "Professional Certificate in AI"
    ]
    
    # KEEPING ALL ORIGINAL DESIGNATIONS
    tech_designations = [
        "Software Engineer", "Software Developer", "Junior Developer", "Associate Engineer",
        "Frontend Developer", "Backend Developer", "Full Stack Developer", "Mobile Developer",
        "Senior Software Engineer", "Senior Developer", "Team Lead", "Technical Lead",
        "DevOps Engineer", "Data Scientist", "Machine Learning Engineer", "AI Engineer",
        "Cloud Engineer", "Security Engineer", "QA Engineer", "Automation Engineer",
        "Principal Engineer", "Staff Engineer", "Engineering Manager", "Tech Lead",
        "Senior Data Scientist", "Lead DevOps Engineer", "Cloud Architect", "Solution Architect",
        "Security Architect", "Data Engineer", "MLOps Engineer", "Platform Engineer",
        "Engineering Director", "Head of Engineering", "CTO", "VP Engineering",
        "Product Manager", "Technical Product Manager", "Program Manager", "Delivery Manager",
        "Blockchain Developer", "IoT Engineer", "AR/VR Developer", "Game Developer",
        "Embedded Systems Engineer", "Firmware Engineer", "Systems Analyst",
        "Database Administrator", "Network Engineer", "Site Reliability Engineer",
        "Performance Engineer", "Release Engineer", "Build Engineer"
    ]
    
    # KEEPING ALL ORIGINAL COLLEGES
    tech_colleges = [
        "NED University of Engineering", "FAST University", "LUMS", "IBA Karachi",
        "GIK Institute", "UET Lahore", "University of Karachi", "KU",
        "COMSATS University", "PUCIT", "NUST", "Air University", "BUITEMS",
        "SZABIST", "IBA Sukkur", "UIT Karachi", "Comsats Islamabad", "UET Peshawar",
        "UET Taxila", "IIUI", "QAU", "GC University Lahore", "UoP", "MUET",
        "UET Faisalabad", "NUST SEECS", "NUST CEME", "NUST MCS", "NUST EME",
        "Bahria University", "SZABIST Karachi", "IMSciences Peshawar",
        "Stanford University", "MIT", "Carnegie Mellon", "UC Berkeley",
        "University of Illinois Urbana-Champaign", "Georgia Tech",
        "University of Washington", "University of Texas at Austin",
        "University of Toronto", "University of Waterloo", "Imperial College London",
        "ETH Zurich", "National University of Singapore", "Tsinghua University",
        "Harvard University", "Caltech", "Princeton University", "Yale University"
    ]
    
    synthetic_resumes = []
    
    for i in range(num_samples):
        # Random selection with Pakistani preference
        name_choice = random.choices([pakistani_names, western_names], weights=[0.6, 0.4], k=1)[0]
        name = random.choice(name_choice)
        
        company_choice = random.choices(
            [pakistani_tech_companies, international_tech_companies, tech_service_companies],
            weights=[0.4, 0.4, 0.2], k=1
        )[0]
        company = random.choice(company_choice)
        
        location = random.choice(all_locations)
        degree = random.choice(tech_degrees)
        college = random.choice(tech_colleges)
        designation = random.choice(tech_designations)
        
        # Generate realistic dates
        current_year = datetime.now().year
        grad_year = random.randint(current_year - 8, current_year - 2)
        experience_years = current_year - grad_year
        
        # Create email from name
        email_name = name.lower().replace(' ', '.').replace("'", "")
        email = f"{email_name}@gmail.com"
        
        # Create clean, structured resume content
        content = f"""{name}
{designation}
{company}

Location: {location}
Email: {email}
Experience: {experience_years} years

PROFESSIONAL SUMMARY
{designation} with {experience_years} years of experience at {company}. Skilled in software development and cloud technologies.

WORK EXPERIENCE
{designation} | {company} | {location}
• Developed scalable applications
• Collaborated with cross-functional teams
• Implemented best practices

EDUCATION
{degree} | {college} | {grad_year}

SKILLS
Programming: Python, Java, JavaScript
Tools: Docker, Kubernetes, AWS"""

        # ✅ FIXED: Create PROPER NER annotations (clean, precise, no overlaps)
        annotations = []
        
        # Helper function to add clean annotation
        def add_clean_annotation(label, text):
            if text and len(text) < 100:  # Only reasonable length entities
                start = content.find(text)
                if start != -1:
                    annotations.append({
                        "label": [label],
                        "points": [{"start": start, "end": start + len(text), "text": text}]
                    })
        
        # Add ONLY clean, precise entities (no paragraph-long annotations)
        add_clean_annotation("Name", name)
        add_clean_annotation("Companies worked at", company)
        add_clean_annotation("Designation", designation)
        add_clean_annotation("Location", location)
        add_clean_annotation("Degree", degree)
        add_clean_annotation("College Name", college)
        add_clean_annotation("Email Address", email)
        
        # Only add resume if we have good annotations
        if len(annotations) >= 4:  # At least name, company, designation, degree
            synthetic_resumes.append({
                "content": content,
                "annotation": annotations,
                "extras": None
            })
        
        # Progress indicator
        if (i + 1) % 200 == 0:
            print(f"Generated {i + 1}/{num_samples} tech resumes...")
    
    return synthetic_resumes

def save_synthetic_data(synthetic_resumes, output_file="tech_resumes.json"):
    """Save synthetic data to JSON file"""
    print(f"💾 Saving {len(synthetic_resumes)} tech resumes to {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for resume in synthetic_resumes:
            f.write(json.dumps(resume, ensure_ascii=False) + '\n')
    
    print(f"✅ Successfully saved {len(synthetic_resumes)} tech resumes")

def combine_datasets(original_file, synthetic_file, output_file="enhanced_tech_resumes.json"):
    """Combine original data with synthetic tech data"""
    print("🔄 Combining datasets...")
    
    combined_data = []
    
    # Load original data
    try:
        with open(original_file, 'r', encoding='utf-8') as f:
            for line in f:
                combined_data.append(json.loads(line.strip()))
        print(f"✅ Loaded {len(combined_data)} original resumes")
    except FileNotFoundError:
        print(f"⚠️ Original file {original_file} not found, using synthetic only")
        return synthetic_file
    
    # Load synthetic data
    try:
        with open(synthetic_file, 'r', encoding='utf-8') as f:
            synthetic_count = 0
            for line in f:
                combined_data.append(json.loads(line.strip()))
                synthetic_count += 1
        print(f"✅ Added {synthetic_count} synthetic resumes")
    except FileNotFoundError:
        print(f"❌ Synthetic file {synthetic_file} not found")
        return original_file
    
    # Save combined data
    with open(output_file, 'w', encoding='utf-8') as f:
        for resume in combined_data:
            f.write(json.dumps(resume, ensure_ascii=False) + '\n')
    
    print(f"🎉 Combined dataset saved: {len(combined_data)} total resumes")
    return output_file

if __name__ == "__main__":
    print("🚀 GENERATING ENHANCED TECH RESUME DATASET")
    print("=" * 50)
    
    # Generate synthetic data
    synthetic_resumes = generate_tech_focused_resumes(num_samples=2000)
    
    # Save synthetic data
    save_synthetic_data(synthetic_resumes, "tech_resumes.json")
    
    # Try to combine with original
    final_file = combine_datasets("entities2.json", "tech_resumes.json")
    
    print("\n📊 DATASET COMPOSITION:")
    print(f"   • Total Companies: {len(pakistani_tech_companies + international_tech_companies + tech_service_companies)}")
    print(f"   • Total Designations: {len(tech_designations)}")
    print(f"   • Total Locations: {len(all_locations)}")
    print(f"   • Total Degrees: {len(tech_degrees)}")
    print(f"   • Total Colleges: {len(tech_colleges)}")
    print(f"   • Total Resumes: {len(synthetic_resumes)}")
    
    print("\n🎯 READY FOR TRAINING!")
    print(f"Use this file: {final_file}")