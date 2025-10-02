from pastpaper_handler import PastPaperProcessor, PastPaperSession

class SimpleDoc:
    def __init__(self, content: str):
        self.page_content = content
        self.metadata = {}

SAMPLE_CONTENT = """
Question 1: What is a firewall?
A. A physical barrier
B. A network security device
C. A web server
D. A VPN

Question 2: Which is a desired attribute of a firewall?
A. Allow all traffic
B. Permit only authorized traffic
C. Be easy to penetrate
D. Block all traffic

Question 3: Primary function of a firewall is to?
A. Speed up network
B. Screen malicious programs/users
C. Encrypt all traffic
D. Provide Wi-Fi

Question 4: Firewalls control traffic by?
A. Randomly allowing or blocking
B. Using rule sets to inspect packets
C. Redirecting traffic
D. Compressing data

Question 5: Crucial design goal for firewall?
A. Be easily bypassed
B. Immune to penetration
C. Use open-source OS
D. Disable during peak hours

Question 6: Next gen firewalls add which capability?
A. Packet coloring
B. Application-layer inspection and IPS
C. Dial-up support
D. Tape backup

Question 7: A stateful firewall tracks?
A. Weather
B. Connection state
C. User passwords
D. Disk space

Question 8: Which is NOT a firewall type?
A. Packet-filtering
B. Circuit-level gateway
C. Proxy server
D. Load balancer

Question 9: Default deny policy means?
A. Allow unless denied
B. Deny unless explicitly allowed
C. Always allow
D. Always deny

Question 10: Placement best practice?
A. Behind all hosts
B. Between internet and DMZ/internal networks
C. On user desktops only
D. In the cloud only

Question 11: Logging is used for?
A. Decoration
B. Auditing and incident response
C. Making packets faster
D. DNS caching

Question 12: NAT on a firewall hides?
A. MAC addresses
B. Internal IPs
C. Passwords
D. URLs
"""

def run_pastpaper_batch_smoke():
    docs = [ SimpleDoc(SAMPLE_CONTENT) ]
    questions = PastPaperProcessor.extract_questions(docs)
    assert len(questions) == 12, f"Expected 12 questions extracted, got {len(questions)}"

    session = PastPaperSession()
    session.start_paper("CSC999", "2024", questions)

    # First batch
    batch1, more1 = session.get_next_batch()
    assert len(batch1) == 5, f"First batch should have 5, got {len(batch1)}"
    assert more1 is True, "Should have more after first batch"

    # Second batch
    batch2, more2 = session.get_next_batch()
    assert len(batch2) == 5, f"Second batch should have 5, got {len(batch2)}"
    assert more2 is True, "Should still have more after second batch"

    # Third (final) batch
    batch3, more3 = session.get_next_batch()
    assert len(batch3) == 2, f"Third batch should have 2, got {len(batch3)}"
    assert more3 is False, "Should not have more after final batch"

    print("OK: Batching returns 5 + 5 + 2 with correct has_more flags.")


# New smoke test: enumerated style with "1)" numbering and subparts
SAMPLE_ENUMERATED_CONTENT = """
1) Using the keywords "good, bad, input, output" clearly distinguish between accuracy and security.

2) Distinguish Threat, Vulnerability and Exploit with examples.

3) What is a Botnet? Outline the life cycle.

4) Distinguish between malware and ransomware with one example each.

5) Data Encryption Standard (DES) is symmetric.
   a. What is Symmetric Key Cryptography?
   b. How many rounds are supported by DES?
   c. What is the Key Size in DES?
   d. What is the Block Size in DES?
   e. In what ways is DES considered weak?

6) What is PKI?

7) What is the purpose of PKI?

8) Distinguish between Block and Stream Ciphers.

9) CBC is an example of a block cipher.
   a. What is CBC mode of encryption?
   b. What is IV in the context of CBC?
   c. What is the problem with a fixed IV?
   d. Explain any general attacks against block ciphers.

10) Describe a Feistel structure.

11) Distinguish between a Digital Signature and a Digital Certificate.

12) Explain how Diffie-Hellman key exchange is implemented in RSA.

13) In the context of hashing:
    a. Explain pre-image resistance of a function H.
    b. Describe HMAC and how it is used.

14) Distinguish between authentication and authorization.

15) Explain the Kerberos authentication scheme.

16) Explain what is meant by a Kerberos realm.

17) Describe the Needham-Schroeder Protocol.

18) What is a Stateful Packet Inspection Firewall?
"""


def run_enumerated_style_smoke():
    docs = [SimpleDoc(SAMPLE_ENUMERATED_CONTENT)]
    questions = PastPaperProcessor.extract_questions(docs)
    assert len(questions) == 18, f"Expected 18 questions extracted for enumerated style, got {len(questions)}\n\nFirst: {questions[0] if questions else 'NONE'}"
    # Spot-check a few questions contain expected fragments
    assert "Question 1:" in questions[0]
    assert "Question 9:" in questions[8]
    assert "Question 18:" in questions[-1]

if __name__ == "__main__":
    run_pastpaper_batch_smoke()
    run_enumerated_style_smoke()
    print("All smoke tests passed.")
