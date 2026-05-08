export const CURRENT_LEGAL_VERSIONS = {
  terms: '2026-05-08',
  privacy: '2026-05-08',
  copyright: '2026-05-08',
};

export const LEGAL_DOCUMENT_ORDER = ['terms', 'privacy', 'copyright'];

export const LEGAL_DOCUMENTS = {
  terms: {
    key: 'terms',
    title: 'Terms of Use',
    version: CURRENT_LEGAL_VERSIONS.terms,
    summary:
      'Draft terms for the private-editor launch. Review with counsel before public production use.',
    sections: [
      {
        title: 'Service Scope',
        body: [
          'AutoTuneSyncer is provided as a private video composition and collaboration tool. Public feeds, follower graphs, and public sharing are not part of this release.',
          'You may use the service only for lawful purposes and only with media, MIDI files, and project materials that you own or are authorized to use.',
        ],
      },
      {
        title: 'Your Content',
        body: [
          'You retain ownership of the content you upload. You grant the service a limited license to store, process, transform, and deliver that content solely to operate the editor, render jobs, project backups, and collaboration features.',
          'You are responsible for securing all permissions, licenses, and consents needed for the audio, MIDI, video, images, and other materials in your projects.',
        ],
      },
      {
        title: 'Prohibited Uses',
        body: [
          'Do not upload infringing content, illegal content, malware, or content that violates another person\'s rights, privacy, or publicity rights.',
          'Do not abuse compute resources, attempt to bypass account limits, or interfere with other users, collaborators, or the platform infrastructure.',
        ],
      },
      {
        title: 'Accounts and Enforcement',
        body: [
          'You are responsible for activity under your account and for keeping your credentials secure.',
          'The operator may remove content, suspend access, or terminate accounts for policy violations, infringement complaints, repeated abuse, or payment failures.',
        ],
      },
      {
        title: 'Disclaimers',
        body: [
          'The service is provided on an as-is and as-available basis. Render times, storage availability, export quality, and collaboration uptime are not guaranteed.',
          'These draft terms do not replace legal advice and should be finalized before launch, especially if billing or public sharing is introduced later.',
        ],
      },
    ],
  },
  privacy: {
    key: 'privacy',
    title: 'Privacy Policy',
    version: CURRENT_LEGAL_VERSIONS.privacy,
    summary:
      'Draft privacy policy describing the minimum data needed to run private accounts, projects, rendering, and collaboration.',
    sections: [
      {
        title: 'Information Collected',
        body: [
          'The service stores account information such as username, email address, password hash, profile settings, and legal acceptance records.',
          'It also stores project data, uploaded clips, background assets, render metadata, collaboration records, and operational logs required to process jobs and diagnose failures.',
        ],
      },
      {
        title: 'How Information Is Used',
        body: [
          'Information is used to authenticate users, operate projects, support collaboration, process renders, enforce policies, prevent abuse, and communicate service notices.',
          'Project data is not used to power a public recommendation feed in this release.',
        ],
      },
      {
        title: 'Sharing and Processors',
        body: [
          'Data may be processed by infrastructure providers, storage providers, payment processors, email providers, or analytics and logging vendors that help run the service.',
          'The operator does not sell project content or personal data. Data may be disclosed when required by law, to enforce rights, or to respond to valid legal notices.',
        ],
      },
      {
        title: 'Retention and Security',
        body: [
          'Account and project data is retained for as long as needed to provide the service, comply with legal obligations, resolve disputes, and enforce agreements.',
          'Reasonable technical and organizational safeguards should be used, but no internet service can promise absolute security.',
        ],
      },
      {
        title: 'User Choices',
        body: [
          'Users should be able to update profile information, change credentials, and request deletion of their account and projects subject to legal and billing retention requirements.',
          'Before public launch, this draft should be expanded with jurisdiction-specific rights, cookie disclosures, and processor-specific contact details.',
        ],
      },
    ],
  },
  copyright: {
    key: 'copyright',
    title: 'Copyright and DMCA Policy',
    version: CURRENT_LEGAL_VERSIONS.copyright,
    summary:
      'Draft copyright policy for a private collaboration tool. Replace placeholders and register a DMCA agent before launch.',
    sections: [
      {
        title: 'Upload Rule',
        body: [
          'Upload only content that you created, own, or are licensed to use. This includes MIDI files, audio, video clips, images, and any other material added to a project.',
          'If you collaborate on a project, each collaborator is responsible for the materials they contribute.',
        ],
      },
      {
        title: 'DMCA Safe Harbor Basics',
        body: [
          'Before public launch, the operator should register a DMCA agent with the U.S. Copyright Office and publish the registered contact details here.',
          'The service should respond expeditiously to valid takedown notices, document repeat infringement, and suspend repeat infringers where appropriate.',
        ],
      },
      {
        title: 'Draft Notice Procedure',
        body: [
          'A copyright notice should identify the copyrighted work, the allegedly infringing material, the reporting party\'s contact information, and the required legal statements under the DMCA.',
          'This draft implementation is incomplete until the operator adds the production notice address, designated agent information, and a documented counter-notice process.',
        ],
      },
      {
        title: 'Product Warnings',
        body: [
          'The editor should warn users that rendered outputs can still infringe if the source media is not properly licensed.',
          'Public rediscovery or embedding should remain disabled unless a separate hosting and takedown workflow is intentionally introduced later.',
        ],
      },
    ],
  },
};

export function getLegalDocument(documentKey) {
  return LEGAL_DOCUMENTS[documentKey] || LEGAL_DOCUMENTS.terms;
}