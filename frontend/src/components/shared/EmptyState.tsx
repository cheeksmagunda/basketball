import styles from './EmptyState.module.css';

interface EmptyStateAction {
  label: string;
  onClick: () => void;
}

interface EmptyStateProps {
  icon?: string;
  message: string;
  action?: EmptyStateAction;
}

export default function EmptyState({ icon, message, action }: EmptyStateProps) {
  return (
    <div className={styles['empty-state']}>
      {icon && <span className={styles.icon}>{icon}</span>}
      <p>{message}</p>
      {action && (
        <button
          type="button"
          className={styles['secondary-btn']}
          onClick={action.onClick}
        >
          {action.label}
        </button>
      )}
    </div>
  );
}
